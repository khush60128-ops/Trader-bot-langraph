"""
config.py — Centralised configuration loader.
All modules import from here; nothing reads os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (works whether you run from any sub-dir)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise EnvironmentError(
            f"❌  Missing required environment variable: {key}\n"
            f"    Please fill it in your .env file at: {_ENV_PATH}"
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip() or default


# ── LLM ───────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = _require("GOOGLE_API_KEY")
GEMINI_MODEL: str = "gemini-3-flash-preview"

# ── Search ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = _require("TAVILY_API_KEY")

# ── Mudrex ────────────────────────────────────────────────────────────────────
MUDREX_API_SECRET: str = _require("MUDREX_API_SECRET")
MUDREX_BASE_URL: str = _optional("MUDREX_BASE_URL", "https://trade.mudrex.com/fapi/v1")

# ── Portfolio / Risk ──────────────────────────────────────────────────────────
PORTFOLIO_SIZE_USD: float = float(_optional("PORTFOLIO_SIZE_USD", "100"))
MAX_RISK_SCORE: int = int(_optional("MAX_RISK_SCORE", "3"))
MUDREX_LEVERAGE: int = 5
MUDREX_STOPLOSS_PCT: float = 0.02
MUDREX_TAKEPROFIT_PCT: float = 0.04

# ── Persistence ───────────────────────────────────────────────────────────────
DB_PATH: str = _optional("DB_PATH", "trading_sessions.db")

# ── Chart Auto-Fetch ──────────────────────────────────────────────────────────
# FinViz chart URL template — {ticker} will be replaced at runtime.
# Works for crypto (BTCUSD, ETHUSD) and stocks (AAPL, TSLA).
FINVIZ_CHART_URL: str = "https://charts2.finviz.com/chart.ashx?t={ticker}&p=d&s=l"
