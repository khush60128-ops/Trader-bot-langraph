"""
nodes.py — All agent nodes for the Updated Trade Executor.

Uses LangGraph HITL (interrupt/resume) for trade confirmation.

Nodes: chart_fetcher, news_researcher, vision_chartist,
       risk_manager, trade_decision (HITL interrupt), trade_executor
"""

from __future__ import annotations

import base64, json, re, urllib.request, mimetypes
from datetime import datetime, timezone

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt

try:
    from langchain_tavily import TavilySearch
    _TAVILY_NEW = True
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    _TAVILY_NEW = False

import config
from state import TradingState, ChartAnalysisOutput, NewsResearchOutput, RiskAssessmentOutput

# ── Shared utilities ─────────────────────────────────────────────────────────

def _gemini(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL, google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )

def _response_to_str(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict): parts.append(item.get("text", ""))
            elif isinstance(item, str): parts.append(item)
        return "\n".join(parts)
    return str(content)

def _extract_json(content) -> dict:
    text    = _response_to_str(content)
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text[:600]}")
    return json.loads(match.group())

def _url_to_base64(url: str) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw          = resp.read()
        content_type = resp.headers.get_content_type() or "application/octet-stream"
    if "text/html" in content_type:
        raise ValueError(f"URL returned HTML page, not an image: {url}")
    return base64.b64encode(raw).decode("utf-8"), content_type

def _local_file_to_base64(path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"): mime = "image/png"
    with open(path, "rb") as f: raw = f.read()
    return base64.b64encode(raw).decode("utf-8"), mime

def _run_tavily(query: str) -> list[dict]:
    if _TAVILY_NEW:
        tool = TavilySearch(api_key=config.TAVILY_API_KEY, max_results=10, search_depth="advanced")
        results = tool.invoke(query)
        if isinstance(results, dict): return results.get("results", [])
        return results if isinstance(results, list) else []
    else:
        tool = TavilySearchResults(api_key=config.TAVILY_API_KEY, max_results=10,
                                   search_depth="advanced", include_raw_content=True, include_answer=True)
        raw = tool.invoke(query)
        return raw if isinstance(raw, list) else []

# ── Node 0: Chart Fetcher ────────────────────────────────────────────────────

def chart_fetcher_node(state: TradingState) -> dict:
    """
    Auto-fetches chart image from FinViz. If a chart_url is already provided,
    uses it directly. If auto-fetch fails, uses interrupt() to ask the user.
    """
    ticker    = state["ticker"]
    chart_url = state.get("chart_url", "")

    if chart_url and chart_url.strip():
        print(f"\n[ChartFetcher] ✅ Using provided chart URL for {ticker}")
        return {"chart_url": chart_url, "chart_auto_fetched": False, "chart_fetch_failed": False}

    finviz_ticker = ticker.upper().replace("-", "")
    if finviz_ticker.endswith("USDT"): finviz_ticker = finviz_ticker[:-1]
    auto_url = config.FINVIZ_CHART_URL.format(ticker=finviz_ticker)
    print(f"\n[ChartFetcher] 🔍 Auto-fetching chart for {ticker} from: {auto_url}")

    try:
        _url_to_base64(auto_url)
        print(f"[ChartFetcher] ✅ Chart image downloaded successfully!")
        return {"chart_url": auto_url, "chart_auto_fetched": True, "chart_fetch_failed": False}
    except Exception as e:
        print(f"[ChartFetcher] ⚠️  Auto-fetch failed: {e}")
        print(f"[ChartFetcher] Requesting chart image from user via interrupt...")

        # ── HITL interrupt: ask user for chart image ─────────────────────────
        user_response: dict = interrupt({
            "action_required": "CHART_IMAGE_NEEDED",
            "ticker": ticker,
            "reason": f"Auto-fetch failed: {e}",
            "message": (
                f"Could not auto-fetch chart for {ticker}. "
                "Please provide a chart image URL or local file path."
            ),
        })
        # Resumes here after Command(resume={"chart_url": "..."})

        user_chart_url = user_response.get("chart_url", "")
        if user_chart_url:
            print(f"[ChartFetcher] ✅ User provided chart: {user_chart_url[:80]}")
            return {"chart_url": user_chart_url, "chart_auto_fetched": False, "chart_fetch_failed": False}
        else:
            print(f"[ChartFetcher] ❌ No chart provided by user.")
            return {"chart_url": "", "chart_auto_fetched": False, "chart_fetch_failed": True}

# ── Node 1: News Researcher ─────────────────────────────────────────────────

def news_researcher_node(state: TradingState) -> dict:
    ticker = state["ticker"]
    print(f"\n[NewsResearcher] Searching news for '{ticker}' ...")
    raw_results = _run_tavily(f"{ticker} price news analysis last 24 hours crypto market")
    news_block = ""
    for i, r in enumerate(raw_results, 1):
        news_block += f"\n[{i}] {r.get('title','No title')}\n{r.get('content','')[:400]}\nSource: {r.get('url','')}\n"
    print(f"[NewsResearcher] Got {len(raw_results)} articles. Scoring with Gemini ...")
    llm = _gemini()
    prompt = [
        SystemMessage(content=(
            "You are a quantitative financial news analyst. "
            "Output ONLY a valid JSON object — no markdown:\n"
            '{"sentiment_score": <float -1.0 to +1.0>, "summary": "<2-3 sentences>", "sources": ["<url>", ...]}\n'
            "sentiment_score: -1.0=extremely bearish, 0.0=neutral, +1.0=extremely bullish"
        )),
        HumanMessage(content=f"Asset: {ticker}\nNews (last 24h):\n{news_block}\n\nProduce your JSON assessment now."),
    ]
    response = llm.invoke(prompt)
    parsed   = _extract_json(response.content)
    output   = NewsResearchOutput(**parsed)
    print(f"[NewsResearcher] Sentiment: {output.sentiment_score:+.3f}")
    return {
        "news_sentiment": output.sentiment_score, "news_summary": output.summary, "news_sources": output.sources,
        "messages": [AIMessage(content=f"**News Researcher** -- {ticker}\nSentiment: {output.sentiment_score:+.3f}\n{output.summary}", name="NewsResearcher")],
    }

# ── Node 2: Vision Chartist ─────────────────────────────────────────────────

def vision_chartist_node(state: TradingState) -> dict:
    ticker    = state["ticker"]
    chart_url = state.get("chart_url", "")
    print(f"\n[VisionChartist] Processing chart for {ticker} ...")
    if chart_url.startswith("data:"):
        header, image_b64 = chart_url.split(",", 1)
        mime_type = header.split(";")[0].replace("data:", "")
    elif chart_url.startswith("http"):
        image_b64, mime_type = _url_to_base64(chart_url)
    else:
        image_b64, mime_type = _local_file_to_base64(chart_url)
    print(f"[VisionChartist] Sending to Gemini Vision ({mime_type}) ...")
    llm = _gemini(temperature=0.05)
    msg = HumanMessage(content=[
        {"type": "text", "text": (
            "You are an expert Chartered Market Technician (CMT). Analyse this candlestick chart. "
            "Output ONLY a valid JSON object — no markdown:\n"
            '{"support_levels": [<float>], "resistance_levels": [<float>], '
            '"rsi_condition": "oversold"|"neutral"|"overbought", '
            '"trend": "uptrend"|"sideways"|"downtrend", "analysis_text": "<detailed narrative>"}'
        )},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        {"type": "text", "text": "Perform your complete technical analysis now."},
    ])
    response = llm.invoke([msg])
    parsed   = _extract_json(response.content)
    output   = ChartAnalysisOutput(**parsed)
    print(f"[VisionChartist] Trend: {output.trend} | RSI: {output.rsi_condition}")
    return {
        "chart_analysis": output.analysis_text, "support_levels": output.support_levels,
        "resistance_levels": output.resistance_levels, "rsi_condition": output.rsi_condition,
        "chart_trend": output.trend,
        "messages": [AIMessage(content=f"**Vision Chartist** -- {ticker}\nTrend: {output.trend} | RSI: {output.rsi_condition}\nSupport: {output.support_levels}\nResistance: {output.resistance_levels}\n\n{output.analysis_text}", name="VisionChartist")],
    }

# ── Node 3: Risk Manager ────────────────────────────────────────────────────

def risk_manager_node(state: TradingState) -> dict:
    ticker = state["ticker"]
    sentiment = state.get("news_sentiment", 0.0)
    news_summary = state.get("news_summary", "")
    chart_analysis = state.get("chart_analysis", "No chart data.")
    trend = state.get("chart_trend", "unknown")
    rsi = state.get("rsi_condition", "unknown")
    support = state.get("support_levels", [])
    resistance = state.get("resistance_levels", [])
    print(f"\n[RiskManager] Evaluating risk for {ticker} ...")
    llm = _gemini()
    prompt = [
        SystemMessage(content=(
            f"You are a senior risk manager for a ${config.PORTFOLIO_SIZE_USD:,.0f} portfolio. "
            "Output ONLY a valid JSON object — no markdown:\n"
            '{"risk_score": <int 1-10>, "signal": "BUY"|"SELL"|"HOLD", "approved": <true|false>, '
            '"rationale": "<explanation>", "position_size_usd": <float>}\n\n'
            f"Rules: risk_score>{config.MAX_RISK_SCORE} -> approved=false, signal=HOLD. "
            f"Max position=${config.PORTFOLIO_SIZE_USD * 0.5:,.0f}. Conflicting sentiment+trend -> HOLD."
        )),
        HumanMessage(content=(
            f"Asset: {ticker}\nSentiment: {sentiment:+.3f}\n{news_summary}\n\n"
            f"Trend: {trend} | RSI: {rsi}\nSupport: {support} | Resistance: {resistance}\n"
            f"{chart_analysis}\n\nProduce risk assessment JSON now."
        )),
    ]
    response = llm.invoke(prompt)
    parsed = _extract_json(response.content)
    output = RiskAssessmentOutput(**parsed)
    if output.risk_score > config.MAX_RISK_SCORE:
        output.approved = False
        output.signal = "HOLD"
        output.position_size_usd = 0.0
        output.rationale = f"[AUTO-REJECTED] Risk={output.risk_score} > threshold={config.MAX_RISK_SCORE}. " + output.rationale
    output.position_size_usd = min(output.position_size_usd, config.PORTFOLIO_SIZE_USD * 0.5)
    status = "APPROVED" if output.approved else "REJECTED"
    print(f"[RiskManager] {status} | Score: {output.risk_score}/10 | Signal: {output.signal}")
    return {
        "risk_score": output.risk_score, "signal": output.signal,
        "position_size_usd": output.position_size_usd, "risk_rationale": output.rationale,
        "risk_approved": output.approved,
        "messages": [AIMessage(content=f"**Risk Manager** -- {ticker}\nScore: {output.risk_score}/10 | Signal: {output.signal} | {status}\nPosition: ${output.position_size_usd:,.2f}\n{output.rationale}", name="RiskManager")],
    }

# ── Node 4: Trade Decision (HITL with interrupt) ────────────────────────────

def trade_decision_node(state: TradingState) -> dict:
    """
    Uses LangGraph interrupt() to pause the graph and present
    the full analysis to the user for approval.

    The interrupt payload contains all analysis data + risk status.
    main.py catches GraphInterrupt, shows the summary, asks the user,
    and resumes with Command(resume={"approved": bool, "comment": str}).
    """
    ticker         = state["ticker"]
    news_sentiment = state.get("news_sentiment", 0.0)
    news_summary   = state.get("news_summary", "N/A")
    chart_trend    = state.get("chart_trend", "unknown")
    rsi_condition  = state.get("rsi_condition", "unknown")
    support        = state.get("support_levels", [])
    resistance     = state.get("resistance_levels", [])
    chart_analysis = state.get("chart_analysis", "N/A")
    risk_score     = state.get("risk_score", 0)
    signal         = state.get("signal", "HOLD")
    position_usd   = state.get("position_size_usd", 0.0)
    risk_rationale = state.get("risk_rationale", "")
    risk_approved  = state.get("risk_approved", False)

    # ── interrupt(): freeze graph, send analysis to caller ───────────────────
    human_response: dict = interrupt({
        "action_required": "TRADE_APPROVAL",
        "ticker":           ticker,
        "news_sentiment":   news_sentiment,
        "news_summary":     news_summary,
        "chart_trend":      chart_trend,
        "rsi_condition":    rsi_condition,
        "support_levels":   support,
        "resistance_levels": resistance,
        "chart_analysis":   chart_analysis[:300],
        "risk_score":       risk_score,
        "signal":           signal,
        "position_size_usd": position_usd,
        "risk_rationale":   risk_rationale,
        "risk_approved":    risk_approved,
    })
    # ── Resumes here after Command(resume={...}) ─────────────────────────────

    approved: bool = bool(human_response.get("approved", False))
    comment:  str  = human_response.get("comment", "")
    verdict = "APPROVED" if approved else "REJECTED"
    print(f"\n[TradeDecision] {verdict} | Comment: {comment or 'none'}")

    return {
        "human_approved": approved,
        "human_comment":  comment,
        "messages": [HumanMessage(
            content=(
                f"**Trade Decision** -- {ticker}\n"
                f"Risk Approved: {risk_approved} | User Approved: {approved}\n"
                f"{comment}"
            ),
            name="TradeDecision",
        )],
    }

# ── Node 5: Trade Executor (Mudrex Futures API) ─────────────────────────────

MUDREX_FUTURES_URL = "https://trade.mudrex.com/fapi/v1/futures"
DEFAULT_LEVERAGE   = config.MUDREX_LEVERAGE

def _signal_to_order_type(signal: str) -> str:
    return "LONG" if signal == "BUY" else "SHORT"

def _estimate_quantity(position_usd: float, price: float, leverage: int) -> float:
    if price <= 0: raise ValueError("price must be > 0")
    return round((position_usd * leverage) / price, 3)

def _get_current_price(symbol: str) -> float:
    symbol_map = {
        "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
        "BNBUSDT": "binancecoin", "XRPUSDT": "ripple", "DOGEUSDT": "dogecoin",
        "ADAUSDT": "cardano", "AVAXUSDT": "avalanche-2",
        "DOTUSDT": "polkadot", "MATICUSDT": "matic-network",
    }
    coin_id = symbol_map.get(symbol.upper())
    if not coin_id:
        raise ValueError(f"Symbol '{symbol}' not in price map. Supported: {list(symbol_map.keys())}")
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    with urllib.request.urlopen(url, timeout=10) as resp:
        price = float(json.loads(resp.read())[coin_id]["usd"])
    print(f"[TradeExecutor] Price {symbol}: ${price:,.2f}")
    return price

def _mudrex_place_order(symbol, signal, position_usd, leverage=DEFAULT_LEVERAGE,
                        stoploss_pct=config.MUDREX_STOPLOSS_PCT,
                        takeprofit_pct=config.MUDREX_TAKEPROFIT_PCT) -> dict:
    order_type = _signal_to_order_type(signal)
    current_price = _get_current_price(symbol)
    quantity = _estimate_quantity(position_usd, current_price, leverage)
    if order_type == "LONG":
        sl = round(current_price * (1 - stoploss_pct), 2)
        tp = round(current_price * (1 + takeprofit_pct), 2)
    else:
        sl = round(current_price * (1 + stoploss_pct), 2)
        tp = round(current_price * (1 - takeprofit_pct), 2)
    payload = {"leverage": leverage, "quantity": quantity, "order_price": current_price,
               "order_type": order_type, "trigger_type": "MARKET",
               "is_takeprofit": True, "takeprofit_price": tp,
               "is_stoploss": True, "stoploss_price": sl, "reduce_only": False}
    headers = {"Content-Type": "application/json", "X-Authentication": config.MUDREX_API_SECRET}
    url = f"{MUDREX_FUTURES_URL}/{symbol}/order?is_symbol"
    print(f"[TradeExecutor] POST {url}\n{json.dumps(payload, indent=2)}")
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    result = resp.json()
    if not resp.ok or not result.get("success", False):
        errors = result.get("errors", [{"text": resp.text}])
        error_text = " | ".join(e.get("text", str(e)) for e in errors)
        raise requests.HTTPError(f"Mudrex error (HTTP {resp.status_code}): {error_text}", response=resp)
    return result

def trade_executor_node(state: TradingState) -> dict:
    ticker = state["ticker"]
    signal = state.get("signal", "HOLD")
    position_usd = state.get("position_size_usd", 0.0)
    mudrex_symbol = ""
    print(f"\n[TradeExecutor] {signal} -- {ticker} @ ${position_usd:,.2f}")
    if signal == "HOLD":
        result = {"success": False, "status": "SKIPPED", "reason": "Signal is HOLD.",
                  "ticker": ticker, "timestamp": datetime.now(timezone.utc).isoformat()}
    else:
        mudrex_symbol = ticker.upper().replace("-", "")
        if mudrex_symbol.endswith("USD") and not mudrex_symbol.endswith("USDT"):
            mudrex_symbol += "T"
        try:
            api_resp = _mudrex_place_order(mudrex_symbol, signal, position_usd)
            result = {**api_resp, "_meta": {"ticker": ticker, "mudrex_symbol": mudrex_symbol,
                      "signal": signal, "order_type": _signal_to_order_type(signal),
                      "position_usd": position_usd, "leverage": DEFAULT_LEVERAGE,
                      "executed_at": datetime.now(timezone.utc).isoformat()}}
            print(f"[TradeExecutor] Order placed: {result.get('data', {})}")
        except requests.HTTPError as e:
            err = e.response.text if e.response else str(e)
            result = {"success": False, "status": "ERROR", "error": err,
                      "ticker": ticker, "signal": signal, "timestamp": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            result = {"success": False, "status": "ERROR", "error": str(e),
                      "ticker": ticker, "signal": signal, "timestamp": datetime.now(timezone.utc).isoformat()}
    return {
        "trade_result": result, "mudrex_symbol": mudrex_symbol,
        "messages": [AIMessage(content=f"**Trade Executor** -- {ticker}\n```json\n{json.dumps(result, indent=2)}\n```", name="TradeExecutor")],
    }
