# Data-Analyst Telegram Bot

An LLM agent that answers data-analysis questions sent to it on Telegram.
Built for IIT Madras Tools in Data Science, Project 1.

## What it does

Message the bot a data-analysis question (inline data, or a pointer to a public
dataset such as MOSPI). The agent works out the answer — fetching data and
running pandas/numpy code in a sandboxed `run_python` tool when needed — and
replies with exactly one JSON object:

```json
{"answer": {"state": "Assam"}, "log_url": "https://<host>/run.jsonl"}
```

- `answer` is shaped exactly as the question asks.
- `log_url` is a public, wget-able JSONL log of every agent step (questions,
  tool calls, tool outputs, final answers) — one JSON object per line.

Multi-turn conversations are supported: per-chat history is kept and the agent
answers the latest message in context.

## Architecture

- `bot.py` — everything:
  - FastAPI app serving `/health` and `/run.jsonl` (the public agent log)
  - background thread long-polling the Telegram Bot API (`getUpdates`)
  - agentic loop over an OpenAI-compatible chat API with a `run_python` tool
    (pandas, numpy, requests, BeautifulSoup, openpyxl available; network on)
  - keep-warm self-ping so the free host never idles out

## Run

```
pip install -r requirements.txt
export BOT_TOKEN=...        # from @BotFather
export AIPIPE_TOKEN=...     # OpenAI-compatible API token
export BASE_URL=https://your-host   # public URL of this service
uvicorn bot:app --host 0.0.0.0 --port 8000
```
