"""
main.py — Production entrypoint for the Updated Trade Executor.

Uses LangGraph HITL (interrupt / Command(resume=...)) at two points:
  1. chart_fetcher  — if auto-fetch fails, asks user for chart image
  2. trade_decision — shows analysis summary, asks user to approve/reject

Flow:
  User provides --ticker → chart auto-fetched → analysis → risk →
  analysis summary shown → user confirms → trade executed

Usage
─────
  python main.py --ticker BTC-USD
  python main.py --ticker AAPL --chart-url https://direct-image.png
  python main.py --list-sessions
  python main.py --inspect <thread-id>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timezone

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

import config
from graph import build_graph


# ── Initial state factory ────────────────────────────────────────────────────

def make_initial_state(ticker: str, chart_url: str = "") -> dict:
    return {
        "messages":          [],
        "ticker":            ticker,
        "chart_url":         chart_url,
        "chart_auto_fetched": False,
        "chart_fetch_failed": False,
        "news_sentiment":    0.0,
        "news_summary":      "",
        "news_sources":      [],
        "chart_analysis":    "",
        "support_levels":    [],
        "resistance_levels": [],
        "rsi_condition":     "",
        "chart_trend":       "",
        "risk_score":        0,
        "signal":            "HOLD",
        "position_size_usd": 0.0,
        "risk_rationale":    "",
        "risk_approved":     False,
        "human_approved":    False,
        "human_comment":     "",
        "trade_result":      {},
        "mudrex_symbol":     "",
        "error":             "",
    }


# ── HITL: handle chart fetch interrupt ───────────────────────────────────────

def _handle_chart_interrupt(payload: dict) -> dict:
    """Ask the user for a chart image when auto-fetch fails."""
    print(f"\n{'═'*65}")
    print(f"  🖼️  CHART IMAGE REQUIRED")
    print(f"{'═'*65}")
    print(f"  {payload.get('message', 'Auto-fetch failed.')}")
    print(f"  Provide a direct image URL (PNG/JPG/WEBP) or local file path.")
    print(f"{'═'*65}")
    chart_input = input("  Chart image URL or path: ").strip()
    return {"chart_url": chart_input}


# ── HITL: handle trade decision interrupt ────────────────────────────────────

def _handle_trade_interrupt(payload: dict) -> dict:
    """
    Show analysis summary and ask user for trade approval.
    - If risk_approved: "Do you want to proceed?"
    - If NOT risk_approved: "⚠ Risk is HIGH. Are you sure?"
    """
    ticker         = payload.get("ticker", "???")
    news_sentiment = payload.get("news_sentiment", 0.0)
    news_summary   = payload.get("news_summary", "N/A")
    chart_trend    = payload.get("chart_trend", "unknown")
    rsi_condition  = payload.get("rsi_condition", "unknown")
    support        = payload.get("support_levels", [])
    resistance     = payload.get("resistance_levels", [])
    chart_analysis = payload.get("chart_analysis", "N/A")
    risk_score     = payload.get("risk_score", 0)
    signal         = payload.get("signal", "HOLD")
    position_usd   = payload.get("position_size_usd", 0.0)
    risk_rationale = payload.get("risk_rationale", "")
    risk_approved  = payload.get("risk_approved", False)

    # Print full analysis summary
    print(f"\n{'═'*65}")
    print(f"  📊  ANALYSIS SUMMARY — {ticker}")
    print(f"{'═'*65}")
    print(f"\n  📰 NEWS ANALYSIS")
    print(f"  {'─'*40}")
    print(f"  Sentiment Score : {news_sentiment:+.3f}")
    print(f"  Summary         : {news_summary}")
    print(f"\n  📈 CHART ANALYSIS")
    print(f"  {'─'*40}")
    print(f"  Trend           : {chart_trend}")
    print(f"  RSI Condition   : {rsi_condition}")
    print(f"  Support Levels  : {support}")
    print(f"  Resistance      : {resistance}")
    ca = str(chart_analysis)[:200]
    if len(str(chart_analysis)) > 200: ca += "..."
    print(f"  Details         : {ca}")
    print(f"\n  ⚖️  RISK ASSESSMENT")
    print(f"  {'─'*40}")
    print(f"  Risk Score      : {risk_score}/10")
    print(f"  Signal          : {signal}")
    print(f"  Position Size   : ${position_usd:,.2f}")
    print(f"  Risk Approved   : {'✅ YES' if risk_approved else '❌ NO'}")
    rr = str(risk_rationale)[:200]
    if len(str(risk_rationale)) > 200: rr += "..."
    print(f"  Rationale       : {rr}")
    print(f"{'═'*65}")

    # Decision based on risk status
    if risk_approved:
        print(f"\n  ✅ Risk is within acceptable range (score: {risk_score}/10)")
        answer = input("  Do you want to proceed with this trade? [y/N]: ").strip().lower()
        approved = answer in {"y", "yes"}
        if approved:
            comment = "User approved trade after reviewing analysis."
            print(f"\n  [Main] ✅ Trade APPROVED.")
        else:
            comment = "User declined trade after reviewing analysis."
            print(f"\n  [Main] ❌ Trade DECLINED.")
    else:
        print(f"\n  ⚠️  WARNING: Risk is HIGH! (score: {risk_score}/10)")
        print(f"  The risk manager has REJECTED this trade.")
        print(f"  Reason: {str(risk_rationale)[:150]}")
        answer = input("\n  ⚠️  Risk is high. Are you sure you want to proceed? [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            approved = True
            comment = f"User overrode risk rejection (score: {risk_score}/10)."
            print(f"\n  [Main] ⚠️  User OVERRODE risk rejection. Proceeding.")
        else:
            approved = False
            comment = f"User accepted risk rejection (score: {risk_score}/10)."
            print(f"\n  [Main] ❌ Trade CANCELLED. Risk too high.")

    return {"approved": approved, "comment": comment}


# ── Session runner ───────────────────────────────────────────────────────────

def run_session(ticker: str, chart_url: str = "") -> str:
    """
    Run a full trading session with HITL interrupt handling.
    Handles up to 2 interrupt points:
      1. chart_fetcher (if auto-fetch fails)
      2. trade_decision (after analysis, for user approval)
    """
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    _print_header(ticker, thread_id)
    initial_state = make_initial_state(ticker, chart_url)

    # ── Phase 1: Run until first interrupt or completion ─────────────────────
    try:
        result = graph.invoke(initial_state, config=thread_config)
        _print_final_state(result)
        return thread_id
    except GraphInterrupt as exc:
        pass  # Handle below

    # ── Handle interrupts in a loop (chart fetch + trade decision) ───────────
    while True:
        # Get the current state to find the interrupt payload
        snapshot = graph.get_state(thread_config)
        if not snapshot or not snapshot.next:
            break

        # The interrupt payload is in the snapshot tasks
        interrupt_payload = {}
        if hasattr(snapshot, 'tasks') and snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    for intr in task.interrupts:
                        interrupt_payload = intr.value if hasattr(intr, 'value') else {}
                        break
                    break

        action = interrupt_payload.get("action_required", "")

        if action == "CHART_IMAGE_NEEDED":
            print(f"\n[Main] ⏸  Graph paused at Chart Fetcher.")
            human_response = _handle_chart_interrupt(interrupt_payload)
        elif action == "TRADE_APPROVAL":
            print(f"\n[Main] ⏸  Graph paused at Trade Decision.")
            human_response = _handle_trade_interrupt(interrupt_payload)
        else:
            print(f"\n[Main] ⏸  Unknown interrupt: {json.dumps(interrupt_payload, indent=2)}")
            human_response = {"approved": False, "comment": "Unknown interrupt."}

        # ── Resume graph with user response ──────────────────────────────────
        try:
            result = graph.invoke(
                Command(resume=human_response),
                config=thread_config,
            )
            _print_final_state(result)
            return thread_id
        except GraphInterrupt:
            # Another interrupt (e.g. trade_decision after chart_fetcher)
            continue

    return thread_id


# ── Session inspection ───────────────────────────────────────────────────────

def inspect_session(thread_id: str) -> None:
    """Print the full saved state for a thread_id from SQLite."""
    graph = build_graph()
    thread_config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(thread_config)
        print(f"\n{'═'*65}")
        print(f"  📂  SESSION STATE — {thread_id}")
        print(f"{'═'*65}")
        values = snapshot.values
        for key, val in values.items():
            if key == "messages":
                print(f"  messages: [{len(val)} messages]")
                for msg in val:
                    role = getattr(msg, "name", type(msg).__name__)
                    print(f"    [{role}]: {str(msg.content)[:120]}")
            elif key == "trade_result" and isinstance(val, dict):
                print(f"  trade_result:")
                for k, v in val.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key:<25}: {val}")
        print(f"{'═'*65}\n")
        print(f"  Next nodes: {snapshot.next}")
    except Exception as e:
        print(f"[Inspect] ❌ Could not load session: {e}")


def list_sessions() -> None:
    """List all thread_ids stored in the SQLite checkpoint database."""
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.execute(
            "SELECT DISTINCT thread_id, MAX(checkpoint_ns) as last_ns "
            "FROM checkpoints GROUP BY thread_id ORDER BY last_ns DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            print("No sessions found in database.")
            return
        print(f"\n{'═'*65}")
        print(f"  📋  SAVED SESSIONS  ({config.DB_PATH})")
        print(f"{'═'*65}")
        for i, (tid, ns) in enumerate(rows, 1):
            print(f"  {i:>3}. {tid}  [{ns}]")
        print(f"{'═'*65}\n")
    except Exception as e:
        print(f"[Sessions] ❌ Could not read DB: {e}")


# ── Resume helper (for separate-process resume) ─────────────────────────────

def resume_session(thread_id: str, approved: bool, comment: str = "") -> None:
    """Resume an interrupted session by thread_id (e.g. from another terminal)."""
    graph = build_graph()
    thread_config = {"configurable": {"thread_id": thread_id}}
    human_response = {"approved": approved, "comment": comment or "Resumed via CLI."}
    label = "✅ APPROVE" if approved else "❌ REJECT"
    print(f"\n[Main] ▶  Resuming thread: {thread_id}")
    print(f"[Main]    Decision: {label} — {human_response['comment']}")
    try:
        result = graph.invoke(Command(resume=human_response), config=thread_config)
        _print_final_state(result)
    except GraphInterrupt:
        print("[Main] ⚠  Graph hit another interrupt after resume.")
    except Exception as e:
        print(f"[Main] ❌ Error during resume: {e}")
        raise


# ── UI helpers ───────────────────────────────────────────────────────────────

def _print_header(ticker: str, thread_id: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  🤖  MULTI-AGENT FINANCIAL INTELLIGENCE SYSTEM (Updated)")
    print(f"{'═'*65}")
    print(f"  Ticker      : {ticker}")
    print(f"  Thread ID   : {thread_id}")
    print(f"  Portfolio   : ${config.PORTFOLIO_SIZE_USD:,.0f}")
    print(f"  Max Risk    : {config.MAX_RISK_SCORE}/10")
    print(f"  Started     : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'═'*65}\n")


def _print_final_state(state: dict) -> None:
    if not state:
        return
    print(f"\n{'═'*65}")
    print("  📊  FINAL RESULT")
    print(f"{'═'*65}")
    fields = [
        "ticker", "news_sentiment", "chart_trend", "rsi_condition",
        "risk_score", "signal", "position_size_usd",
        "risk_approved", "human_approved", "human_comment",
    ]
    for key in fields:
        val = state.get(key, "—")
        if isinstance(val, float):
            if "sentiment" in key:
                print(f"  {key:<25}: {val:+.4f}")
            else:
                print(f"  {key:<25}: {val:,.2f}")
        else:
            print(f"  {key:<25}: {val}")
    trade = state.get("trade_result", {})
    if trade:
        print(f"\n  Trade Result:")
        for k, v in trade.items():
            print(f"    {k:<22}: {v}")
    error = state.get("error", "")
    if error:
        print(f"\n  ⚠ Error: {error}")
    print(f"{'═'*65}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Financial Intelligence System (Updated)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # New session
    parser.add_argument("--ticker", default="BTC-USD",
                        help="Asset ticker symbol (default: BTC-USD)")
    parser.add_argument("--chart-url", default="",
                        help="Optional: chart image URL or local path (auto-fetched if omitted)")

    # Resume existing session
    parser.add_argument("--resume", metavar="THREAD_ID",
                        help="Resume an interrupted session by thread ID")
    parser.add_argument("--approve", dest="approve", action="store_true", default=True,
                        help="Approve the trade when resuming (default)")
    parser.add_argument("--reject", dest="approve", action="store_false",
                        help="Reject the trade when resuming")
    parser.add_argument("--comment", default="",
                        help="Human comment for resume decision")

    # Inspection
    parser.add_argument("--inspect", metavar="THREAD_ID",
                        help="Print full saved state for a thread ID")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List all thread IDs stored in the checkpoint DB")

    args = parser.parse_args()

    if args.list_sessions:
        list_sessions()
    elif args.inspect:
        inspect_session(args.inspect)
    elif args.resume:
        resume_session(
            thread_id=args.resume,
            approved=args.approve,
            comment=args.comment,
        )
    else:
        thread_id = run_session(
            ticker=args.ticker,
            chart_url=args.chart_url,
        )
        print(f"\n[Main] 💾 Session saved. Thread ID: {thread_id}")
        print(f"[Main] To resume:  python main.py --resume {thread_id} --approve")
        print(f"[Main] To inspect: python main.py --inspect {thread_id}")


if __name__ == "__main__":
    main()
