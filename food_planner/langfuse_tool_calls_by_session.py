
from __future__ import annotations
import os 

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from langfuse import get_client

# Path to the evaluation queries -- containing session_id
EVAL_QUERY_PATH = os.path.join(os.path.dirname(__file__), '../../data/culinary_agent_queries_200_with_ids.csv')

TOOL_CALL_LIST_PATH = os.path.join(os.path.dirname(__file__), './eval_results/culinary_agent_queries_200_with_ids_trace_lists.csv')

# the desired order of tool calling for the agent
DESIRED_TOOL_CALL_ORDER = ['Function: get_local_recipe_type', 
                           'Function: fetch_local_recipe', 
                           'Function: search_web', 
                           'Function: check_cfia_recalls', 
                           'Function: modify_recipe', 
                           'Function: prepare_shopping_list']

@dataclass(frozen=True)
class ToolCall:
    session_id: str
    trace_id: str
    observation_id: str
    tool_name: str
    start_time: Optional[datetime]
    input: Any
    output: Any


def _normalize_type(t: Optional[str]) -> str:
    return (t or "").strip().lower()


def _safe_dt(x) -> Optional[datetime]:
    # Langfuse returns datetimes (pydantic) in many cases; keep as-is if already datetime.
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    # Check if ISO strings:
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except Exception:
        return None


def list_traces_for_session(langfuse, session_id: str, limit: int = 100) -> List[str]:
    """
    Returns trace IDs for a given session_id.
    """
    trace_ids: List[str] = []

    cursor = None
    page = 1

    while True:
        try:
            resp = langfuse.api.trace.list(
                session_id=session_id,   
                limit=limit,
                cursor=cursor,           
            )
        except TypeError:
            resp = langfuse.api.trace.list(
                session_id=session_id,
                limit=limit,
                page=page,
            )

        data = getattr(resp, "data", None) or []
        for t in data:
            trace_ids.append(t.id)

        meta = getattr(resp, "meta", None)
        next_cursor = None
        if meta is not None:
            next_cursor = getattr(meta, "cursor", None) if meta is not None else None

        if next_cursor:
            cursor = next_cursor
            continue

        if len(data) < limit:
            break
        page += 1

    return trace_ids


def extract_agent_tool_calls_for_session(
    session_id: str,
    *,
    limit: int = 100,
) -> List[ToolCall]:
    """
    Fetches all traces for a session, loads their observations, and returns agent tool (function) calls in time order.
    """
    langfuse = get_client() 

    trace_ids = list_traces_for_session(langfuse, session_id=session_id, limit=limit)

    tool_calls: List[ToolCall] = []

    for trace_id in trace_ids:
        # Fetch the full trace including observations 
        trace = langfuse.api.trace.get(trace_id)

        observations = getattr(trace, "observations", None) or []
        for obs in observations:
            tool_name = (getattr(obs, "name", None) or "").strip() or "<unnamed_tool>"
            start_time = _safe_dt(getattr(obs, "start_time", None) or getattr(obs, "startTime", None))

            if 'function' in tool_name.lower():

                tool_calls.append(
                    ToolCall(
                        session_id=session_id,
                        trace_id=trace_id,
                        observation_id=getattr(obs, "id", ""),
                        tool_name=tool_name,
                        start_time=start_time,
                        input=getattr(obs, "input", None),
                        output=getattr(obs, "output", None),
                    )
                )

    # Sort across all traces in the session by time 
    tool_calls.sort(key=lambda x: (x.start_time is None, x.start_time, x.trace_id, x.observation_id))
    return tool_calls


def tool_names_in_order(session_id: str) -> List[str]:
    calls = extract_agent_tool_calls_for_session(session_id)
    return [c.tool_name for c in calls]


def plot_tool_traces(trace_df):
    """
    Plots tool traces for visual evaluation.
    trace_df: list of pandas DataFrames (each one = one run/session)
    tool_col: column in df that contains tool names
    labels: list of labels for legend (same length as dfs), optional
    """
    desired_tool_call_order_to_y = {name: i for i, name in enumerate(DESIRED_TOOL_CALL_ORDER)}

    # Create colors 
    cmap = plt.cm.Blues_r  
    n = len(trace_df)
    norm = mpl.colors.Normalize(vmin=0, vmax=max(n - 1, 1))


    fig, ax = plt.subplots(figsize=(9, 4))

    for i, row in trace_df.iterrows():
        print(f"Processing seesion {i+1}")
        tool_list = ast.literal_eval(row["tool_name_list"])
        y = [desired_tool_call_order_to_y.get(t, 0) for t in tool_list]
        color = cmap(norm(i))
        ax.plot(range(len(y)), y, marker="o", color=color, alpha=0.65, linewidth=2, label=row["session_id"])
        
        
    ax.set_xlabel("Step index")
    ax.set_ylabel("Tool (ordered)")
    ax.set_yticks(range(len(DESIRED_TOOL_CALL_ORDER)))
    ax.set_yticklabels(DESIRED_TOOL_CALL_ORDER)
    # ax.grid(True, axis="y", alpha=0.3)
    # ax.legend(fontsize=13)

    # if title:
    #     ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), './eval_results/culinary_agent_queries_200_with_ids_eval_results.png'))
    return fig, ax


def _main():
    """
    Extracts agent tool calls for all queries and save the list in a csv file."""

    query_df = pd.read_csv(EVAL_QUERY_PATH,
                           usecols=["query_id", "query"],
                           dtype={"query_id": "string", "query": "string"})
    
    
    tool_calls = pd.DataFrame(columns=["session_id", "tool_name_list"]).astype({
        "session_id": "string",
        "tool_name_list": "object", 
    })

    for i, row in enumerate(query_df.itertuples(index=False)):
        print(row)
        session_id = row.query_id
        calls = extract_agent_tool_calls_for_session(session_id)
        tool_calls_list = [c.tool_name for c in calls]
        tool_calls.loc[len(tool_calls)] = [session_id, 
                                           tool_calls_list]

        if i % 10 == 0:
            tool_calls.to_csv(
                TOOL_CALL_LIST_PATH, 
                index=False,
            )

    # plot_tool_traces(tool_calls)



if __name__ == "__main__":
    trace_df = pd.read_csv(TOOL_CALL_LIST_PATH)
    plot_tool_traces(trace_df)