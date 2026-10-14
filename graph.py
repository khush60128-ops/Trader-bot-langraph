"""
graph.py - StateGraph compilation, routing logic, and SqliteSaver persistence.

Updated Topology (with HITL interrupt points)
----------------------------------------------
                      START
                        |
                  chart_fetcher    ◄── interrupt() if auto-fetch fails
                      /     \\
        (parallel)  /       \\  (parallel)
                   v         v
         news_researcher  vision_chartist
                   \\       /
                    \\     /
              (fan-in join)
                     v
               risk_manager
                     |
               trade_decision  ◄── interrupt() shows analysis, asks user
                     |
             human_approved?
            NO |          | YES
             END    trade_executor
                          |
                         END
"""

from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ModuleNotFoundError:
    from langgraph_checkpoint_sqlite import SqliteSaver

import config
from state import TradingState
from nodes import (
    chart_fetcher_node,
    news_researcher_node,
    vision_chartist_node,
    risk_manager_node,
    trade_decision_node,
    trade_executor_node,
)


# ── Routing functions ────────────────────────────────────────────────────────

def _route_human(state: TradingState) -> str:
    """After trade_decision: execute trade or terminate."""
    if state.get("human_approved", False):
        return "trade_executor"
    print("\n[Graph] User declined trade. Terminating session.")
    return END


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    """
    Compile the LangGraph with:
    - chart_fetcher (with HITL interrupt if auto-fetch fails)
    - Parallel fan-out (news + chart analysis)
    - Fan-in to risk manager
    - trade_decision (HITL interrupt for user approval)
    - Conditional routing to trade_executor
    - SqliteSaver for cross-process state persistence
    """
    builder = StateGraph(TradingState)

    # Register all nodes
    builder.add_node("chart_fetcher",   chart_fetcher_node)
    builder.add_node("news_researcher", news_researcher_node)
    builder.add_node("vision_chartist", vision_chartist_node)
    builder.add_node("risk_manager",    risk_manager_node)
    builder.add_node("trade_decision",  trade_decision_node)
    builder.add_node("trade_executor",  trade_executor_node)

    # START → chart_fetcher
    builder.add_edge(START, "chart_fetcher")

    # chart_fetcher → parallel fan-out to both analysis nodes
    builder.add_edge("chart_fetcher", "news_researcher")
    builder.add_edge("chart_fetcher", "vision_chartist")

    # Fan-in: both analysis nodes → risk_manager
    builder.add_edge("news_researcher", "risk_manager")
    builder.add_edge("vision_chartist", "risk_manager")

    # risk_manager → trade_decision (always — trade_decision handles both cases)
    builder.add_edge("risk_manager", "trade_decision")

    # trade_decision → trade_executor | END
    builder.add_conditional_edges(
        "trade_decision",
        _route_human,
        {"trade_executor": "trade_executor", END: END},
    )

    # trade_executor → END
    builder.add_edge("trade_executor", END)

    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    print(f"[Graph] Compiled. DB: {config.DB_PATH} | Max risk: {config.MAX_RISK_SCORE}/10")
    return graph


if __name__ == "__main__":
    g = build_graph()
    print("\n--- LangGraph Topology ---")
    try:
        print(g.get_graph().draw_ascii())
    except Exception as e:
        print(g.get_graph().draw_mermaid())

