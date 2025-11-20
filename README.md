# AI-Driven Crypto Futures Signal Bot

High-confidence trading signal system for USDT perpetuals. The stack combines FastAPI, CCXT, pandas-ta, and an LLM (OpenAI or Anthropic) to synthesize multi-timeframe market context and deliver position recommendations to Discord. Built with the Codex 5 (vibe coding) toolchain.

## Table of Contents
- Overview
- Architecture
- Data & Signal Flow
- Tech Stack
- Setup
- Configuration
- Running Locally
- Discord Usage
- Testing
- Deployment (systemd)
- Notes

## Overview
- Purpose: Generate disciplined LONG/SHORT/NEUTRAL calls only when probability is high.
- Inputs: 15m and 1h OHLCV, SMA(7/25/99), RSI14, Bollinger (20, 2), MACD, order book imbalance (±0.5%), funding rate, and open interest change.
- Outputs: LLM-enforced JSON with decision, confidence, entry/SL/TP, and R:R, rendered to Discord embeds.
- Guardrails: Confidence threshold, SL/TP ordering, and R:R between 1 and 3; otherwise NEUTRAL.

## Architecture
1) Discord Bot: Slash command `/position` forwards the request. Users provide a coin ticker (e.g., `eth`, `btc`) and optional SL%/LLM model choice.
2) FastAPI Backend:
   - Fetches live Bybit data via CCXT.
   - Computes indicators with pandas/pandas-ta.
   - Builds LLM payload and calls OpenAI/Anthropic with a strict JSON schema.
   - Sanitizes/validates the response before returning.
3) LLM Prompt: Stored in `prompt/system_prompt.txt`; can be changed via `.env` or file path.
4) Persistence: Stateless; uses environment variables for secrets and configuration.

## Data & Signal Flow
```
Discord /position -> FastAPI /signal -> CCXT (Bybit) fetch OHLCV/orderbook/OI/funding
                   -> Feature calc (SMA/RSI/BB/MACD) -> LLM (JSON output)
                   -> Guardrails (confidence, SL/TP bounds, R:R) -> Discord embed
```

## Tech Stack
- Python 3.12+, FastAPI, uvicorn
- CCXT, pandas, pandas-ta
- OpenAI / Anthropic SDKs
- Discord.py (slash commands)
- python-dotenv, Pydantic Settings
- Tests: pytest

## Setup
```bash
cd /opt
git clone https://github.com/M00M00M00/ai_futures_position_recommend_bot.git
cd ai_futures_position_recommend_bot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
Copy the template and fill secrets:
```bash
cp .env.example .env
chmod 600 .env
```
Key variables:
- `BYBIT_API_KEY`, `BYBIT_API_SECRET`
- `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`)
- `DISCORD_BOT_TOKEN`
- `LLM_MODEL_NAME` (default `gpt-5-mini`), `LLM_PROVIDER` (`openai` or `anthropic`)
- `LLM_SYSTEM_PROMPT_FILE` (default `prompt/system_prompt.txt`)
- `CONFIDENCE_THRESHOLD` (default 70)
- `BYBIT_TEST_SYMBOL` (default `ETH/USDT:USDT`)
- `DEFAULT_SL_PCT` (default `1.0`)
- `PERP_SUFFIX` (default `USDT` for symbol normalization)

System prompt: edit `prompt/system_prompt.txt` to adjust persona/logic without code changes.

## Running Locally
FastAPI:
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Discord bot:
```bash
source .venv/bin/activate
python bot/main.py
```

Health check:
```bash
curl -s http://127.0.0.1:8000/health
```

## Discord Usage
- Command: `/position`
- Arguments:
  - `symbol`: coin ticker (e.g., `eth`, `btc`). Auto-normalized to `BASE/USDT:USDT`.
  - `sl`: stop loss percent (default 1.0).
  - `model`: optional LLM choice (`gpt-5.1`, `gpt-5-mini` default, `gpt-5-nano`).
- Output: Embed with decision color (green/red/grey), confidence, entry, SL, TP, R:R, analysis, disclaimer.

## Testing
```bash
source .venv/bin/activate
pytest
```
Tests cover data aggregation, signal guardrails, API integration with fakes, and Discord embed formatting (including truncation to Discord limits).

## Deployment (systemd example, optional dedicated user `aifutures`)
Create user and set permissions:
```bash
sudo useradd -r -s /usr/sbin/nologin aifutures || true
sudo chown -R aifutures:aifutures /opt/ai_futures_position_recommend_bot
```

`/etc/systemd/system/aifutures-api.service`:
```
[Unit]
Description=AI Futures FastAPI Service
Wants=network-online.target
After=network-online.target

[Service]
User=aifutures
Group=aifutures
WorkingDirectory=/opt/ai_futures_position_recommend_bot
Environment="PATH=/opt/ai_futures_position_recommend_bot/.venv/bin"
ExecStart=/opt/ai_futures_position_recommend_bot/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/aifutures-bot.service`:
```
[Unit]
Description=AI Futures Discord Bot
Wants=network-online.target
After=network-online.target

[Service]
User=aifutures
Group=aifutures
WorkingDirectory=/opt/ai_futures_position_recommend_bot
Environment="PATH=/opt/ai_futures_position_recommend_bot/.venv/bin"
ExecStart=/opt/ai_futures_position_recommend_bot/.venv/bin/python bot/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable aifutures-api.service aifutures-bot.service
sudo systemctl start aifutures-api.service aifutures-bot.service
sudo systemctl status aifutures-api.service aifutures-bot.service
```

## Notes
- Order book depth uses ±0.5% window; adjust in `app/data.py` if needed.
- Confidence/R:R/SL bounds are enforced post-LLM to prevent unsafe outputs.
- Keep `.env` private; never commit secrets.
