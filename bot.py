"""Data-analyst Telegram bot — TDS Project 1.

An LLM agent that answers data-analysis questions sent over Telegram.
Replies to every message with exactly one JSON object:
    {"answer": <shaped as the question asks>, "log_url": "<public JSONL log>"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat with a
    run_python tool) until the model produces the final JSON answer.
  - A keep-warm thread pings our own public URL so the free host never idles out.
"""

import io
import json
import os
import re
import threading
import time
import traceback
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from dotenv import load_dotenv

load_dotenv()



# ---------------------------------------------------------------- config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

MAX_AGENT_STEPS = 5
PY_TIMEOUT = 15  # seconds for one run_python call
ANSWER_BUDGET = 45# wall-clock seconds before we force a final answer

_log_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}  # chat_id -> chat-completion messages
_hist_lock = threading.Lock()


# ---------------------------------------------------------------- logging
def log_event(**fields):
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- tools
def run_python(code: str) -> str:
    """Execute Python code, return captured stdout (or the error)."""
    out = io.StringIO()
    result: dict = {}

    def target():
        env = {"__name__": "__main__"}
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, env)
            result["ok"] = True
        except Exception:
            result["ok"] = False
            out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)

    if t.is_alive():
     return f"ERROR: code timed out after {PY_TIMEOUT}s"

    text = out.getvalue().strip()

    if text:
      return text[-8000:]

    return (
      "The code executed successfully but produced no stdout. "
      "Use print(...) to display the final result."

    )

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "pandas, numpy, requests, bs4, openpyxl are installed and the "
                "network is available (download public datasets with requests). "
                "Always print() what you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are an expert data-analyst agent answering questions sent to a Telegram bot.

Rules:
1. Work out the answer to the user's LATEST message. Earlier messages in the chat are context for multi-turn tasks.

2. The message may embed data inline or reference a public dataset (e.g. MOSPI, WHO, USGS, data.gov.in). Use the run_python tool whenever external data or computation is required. Do not guess results that can be computed from data. If fetching fails, inspect the response and adapt the approach or use an alternative public source instead of guessing. Only answer from reliable knowledge if the question is purely factual and does not require computation.

3. The message usually specifies the exact JSON shape it expects, for example:
{"answer": {"state": "<state>"}, "log_url": "..."}
Reply using ONLY that JSON object.

4. When you are ready to answer, reply with ONLY the required JSON object. Do not include any prose, explanations, markdown, or code fences. Use the placeholder "LOG_URL" for the log_url value; the harness will replace it with the real URL. Match the requested JSON structure exactly, including keys, nesting, and data types.

5. If the message does not specify a JSON format, reply:
{"answer": <your concise answer>, "log_url": "LOG_URL"}

6. If a mid-conversation message is only setup or context (for example, "I will send data next"), reply:
{"answer": "ok", "log_url": "LOG_URL"}
unless the message asks an actual question.

7. Round numbers exactly as instructed. If no rounding is specified, use reasonable precision. Never add extra keys inside the "answer" object.

8. If a tool call fails or times out, do not blindly repeat the same code. First inspect the error and the API response. If appropriate, modify the approach, adapt the code, or use an alternative public source.

9. Never repeat an identical tool call after it has already failed. Modify the approach based on the observed error instead of blindly retrying.

10. Use the run_python tool only when computation or external data access is actually required. Do not use it for simple factual questions or trivial reasoning.

11. Before accessing JSON fields from any HTTP API, always inspect the response. Never assume keys such as "value" exist. If needed, print response.status_code, the response headers, the top-level JSON keys, or the first part of the response body before processing it.

12. If an external API returns an unexpected structure or parsing fails, do not repeat the same request with nearly identical code. Inspect the response, adapt the code, or use another reliable public source if available.

13. Prefer stable data formats such as CSV or official downloadable files whenever they are available instead of complex JSON APIs.

14. When writing Python that fetches external data:
- Check response.status_code.
- Validate the JSON structure before accessing fields.
- Handle missing keys safely.
- Verify that required columns or fields exist before processing.
- Print useful diagnostics instead of crashing.

15. Keep Python code efficient and minimal. Avoid unnecessary libraries or repeated downloads. Perform only the computation needed to answer the question.

16. After obtaining the required result, immediately produce the final JSON response. Do not perform unnecessary additional tool calls.
"""


# ---------------------------------------------------------------- llm
def chat_completion(messages, use_tools=True):
    body = {"model": MODEL, "messages": messages, "temperature": 0}
    if use_tools:
        body["tools"] = TOOLS
    r = requests.post(
        f"{MODEL_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (data-analyst-bot)",
        },
        json=body,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def extract_json(text: str):
    """Pull the first balanced JSON object out of model text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        # keep the last 20 turns
        del history[:-20]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)

    final_text = None
    deadline = time.time() + ANSWER_BUDGET
    for step in range(MAX_AGENT_STEPS):
        out_of_time = time.time() > deadline
        if out_of_time:
            messages.append(
                {
                    "role": "user",
                    "content": "Time is up. Reply NOW with only your best final JSON object.",
                }
            )
        try:
            msg = chat_completion(messages, use_tools=not out_of_time)
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, error=str(e))
            time.sleep(2)
            try:
                msg = chat_completion(messages, use_tools=True)
            except Exception as e2:
                log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                break
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except json.JSONDecodeError:
                    code = tc["function"]["arguments"]
                log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:4000])
                output = run_python(code)
                log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:4000])
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": output}
                )
            continue
        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text) if final_text else None
    if obj is None:
        obj = {"answer": (final_text or "unable to determine").strip()[:1000]}
    if "answer" not in obj:
        obj = {"answer": obj}
    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False)

    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    log_event(event="answer", chat_id=chat_id, reply=reply)
    return reply


# ---------------------------------------------------------------- telegram
def tg(method, **params):
    r = requests.post(f"{TG_API}/{method}", json=params, timeout=65)
    return r.json()


def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return
    text = msg.get("text") or msg.get("caption") or ""
    chat_id = msg["chat"]["id"]
    if not text:
        return
    try:
        reply = solve(chat_id, text)
    except Exception:
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    tg("sendMessage", chat_id=chat_id, text=reply)


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, model=MODEL)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            resp = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            ).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    """Ping our own public URL so a free host never spins down."""
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------- web app
app = FastAPI()


@app.on_event("startup")
def _start():
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "model": MODEL, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.get("/")
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}
