"""
state.py — TradingState TypedDict + all Pydantic output schemas.
"""

from __future__ import annotations

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ══════════════════════════════════════════════════════════════════════════════
#  Core graph state
# ══════════════════════════════════════════════════════════════════════════════

class TradingState(TypedDict):
    """
    Single shared state object threaded through every node.

    All fields use their Python-native types; LangGraph serialises this to
    SQLite automatically via the SqliteSaver checkpointer.
    """

    # Full conversation / agent message history (append-only via add_messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Input ─────────────────────────────────────────────────────────────────
    ticker: str            # e.g. "BTC-USD", "AAPL"
    chart_url: str         # public URL or base64 data-URI of the chart image

    # ── Chart fetch status ────────────────────────────────────────────────────
    chart_auto_fetched: bool   # True if chart was auto-fetched, False if user-provided
    chart_fetch_failed: bool   # True if auto-fetch failed and user needs to provide

    # ── Node 1 output ─────────────────────────────────────────────────────────
    news_sentiment: float                  # [-1.0, +1.0]
    news_summary: str
    news_sources: list[str]

    # ── Node 2 output ─────────────────────────────────────────────────────────
    chart_analysis: str
    support_levels: list[float]
    resistance_levels: list[float]
    rsi_condition: str                     # oversold | neutral | overbought
    chart_trend: str                       # uptrend | sideways | downtrend

    # ── Node 3 output ─────────────────────────────────────────────────────────
    risk_score: int
    signal: Literal["BUY", "SELL", "HOLD"]
    position_size_usd: float
    risk_rationale: str
    risk_approved: bool

    # ── Node 4 output (trade decision) ────────────────────────────────────────
    human_approved: bool
    human_comment: str

    # ── Node 5 output ─────────────────────────────────────────────────────────
    mudrex_symbol: str
    trade_result: dict

    # ── Error tracking ────────────────────────────────────────────────────────
    error: str


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic schemas for structured LLM output
# ══════════════════════════════════════════════════════════════════════════════

class NewsResearchOutput(BaseModel):
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Aggregate sentiment: -1=extremely bearish, +1=extremely bullish"
    )
    summary: str = Field(..., description="2-3 sentence summary of key headlines")
    sources: list[str] = Field(default_factory=list, description="Source URLs used")


class ChartAnalysisOutput(BaseModel):
    support_levels: list[float] = Field(..., description="Key price support levels")
    resistance_levels: list[float] = Field(..., description="Key price resistance levels")
    rsi_condition: Literal["oversold", "neutral", "overbought"]
    trend: Literal["uptrend", "sideways", "downtrend"]
    analysis_text: str = Field(..., description="Full technical analysis narrative")


class RiskAssessmentOutput(BaseModel):
    risk_score: int = Field(..., ge=1, le=10, description="1=very low risk, 10=extreme risk")
    signal: Literal["BUY", "SELL", "HOLD"]
    approved: bool = Field(..., description="True = trade may proceed to human gate")
    rationale: str = Field(..., description="Reasoning behind the decision")
    position_size_usd: float = Field(..., description="Recommended position in USD")
