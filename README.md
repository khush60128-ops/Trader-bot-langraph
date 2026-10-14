# Multi-Agent Financial Intelligence System (Updated)

Production-grade LangGraph system with **Gemini 2.0 Flash** (text + vision),
**Tavily** real-time news search, **Mudrex** live trade execution, and
**interactive risk-aware trade confirmation**.

---

## What's New (vs Original)

| Feature | Original | Updated |
|---------|----------|---------|
| Chart input | `--chart-url` required | Auto-fetched from FinViz; asks user only if it fails |
| Risk flow | Risk rejected → session ends | Risk rejected → warns user, asks "Are you sure?" |
| Human gate | `interrupt()` + resume via CLI | Inline interactive prompt (no restart needed) |
| Trade confirm | Separate approve/reject step | Single prompt after analysis summary |

---

## Project Structure

```
TradeExecutorUpdated/
├── .env                ← Fill in your API keys here
├── config.py           ← Settings + FinViz chart URL template
├── state.py            ← TradingState TypedDict + Pydantic schemas
├── nodes.py            ← All 6 agent node implementations
├── graph.py            ← StateGraph wiring + SqliteSaver
├── main.py             ← CLI entrypoint
└── requirements.txt
```

---

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Fill in `.env`

```env
GOOGLE_API_KEY=AIza...
TAVILY_API_KEY=tvly-...
MUDREX_API_SECRET=...
MUDREX_BASE_URL=https://trade.mudrex.com/fapi/v1
PORTFOLIO_SIZE_USD=10000
MAX_RISK_SCORE=7
DB_PATH=trading_sessions.db
```

---

## Running

### Basic — just provide a ticker (chart auto-fetched)

```bash
python main.py --ticker BTC-USD
```

### With explicit chart image

```bash
python main.py --ticker AAPL --chart-url https://your-direct-image.png
python main.py --ticker ETH-USD --chart-url C:\charts\eth.png
```

### Session management

```bash
python main.py --list-sessions
python main.py --inspect <thread-id>
```

---

## Flow

```
User provides TICKER
        │
        ▼
  Auto-fetch chart (FinViz)
        │
   Image OK? ──NO──► Ask user for chart path/URL
        │
       YES
        │
        ▼
  ┌──────────────────────┐
  │ news_researcher      │ ─┐
  │ vision_chartist      │ ─┤ parallel
  └──────────────────────┘  │
                      fan-in
                          │
                          ▼
                   risk_manager
                          │
           ┌── ANALYSIS SUMMARY ──┐
           │  (printed to user)   │
           └──────────────────────┘
                          │
                 risk_approved?
                YES │         │ NO
                    │    "⚠ Risk HIGH!"
                    │    "Are you sure?"
                    │     NO │    │ YES
                    │      END   │
                    ▼            ▼
             "Want to trade?"
              YES │    │ NO
                  │   END
                  ▼
           trade_executor
                  │
                 END
```

---

## State Schema

```python
class TradingState(TypedDict):
    messages:           Annotated[Sequence[BaseMessage], add_messages]
    ticker:             str
    chart_url:          str
    chart_auto_fetched: bool
    chart_fetch_failed: bool
    news_sentiment:     float
    news_summary:       str
    news_sources:       list[str]
    chart_analysis:     str
    support_levels:     list[float]
    resistance_levels:  list[float]
    rsi_condition:      str
    chart_trend:        str
    risk_score:         int
    signal:             Literal["BUY", "SELL", "HOLD"]
    position_size_usd:  float
    risk_rationale:     str
    risk_approved:      bool
    human_approved:     bool
    human_comment:      str
    trade_result:       dict
    error:              str
```
