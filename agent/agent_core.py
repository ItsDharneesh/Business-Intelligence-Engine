from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agent.schema_catalog import (
    constrain_run_bi_query_arguments,
    detect_health_intent,
    get_planner_schema_context,
    infer_time_scope_from_query,
)
from agent.tool_registry import TOOLS
from config import OPENAI_API_KEY
from utils.trace_logger import log_trace

client = OpenAI(api_key=OPENAI_API_KEY)


def _sanitize_tool_args(tool_name: str, arguments: dict[str, Any], user_query: str) -> dict[str, Any]:
    args = dict(arguments or {})

    if tool_name == "run_bi_query":
        args, warnings = constrain_run_bi_query_arguments(args, user_query)
        for warning in warnings:
            log_trace(f"Planner constraint: {warning}")

    if tool_name == "fetch_deals":
        if "time_scope" not in args or not str(args.get("time_scope", "")).strip():
            args["time_scope"] = infer_time_scope_from_query(user_query)

    if tool_name == "assess_business_health":
        if "time_scope" not in args or not str(args.get("time_scope", "")).strip():
            args["time_scope"] = infer_time_scope_from_query(user_query)

    return args


def _build_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for name, tool in TOOLS.items()
    ]


def _call_llm_for_tool_selection(user_query: str, tool_schemas: list[dict[str, Any]]) -> Any:
    schema_context = get_planner_schema_context()
    system_prompt = """
You are a production Business Intelligence planning agent connected to live Monday.com data.

Core behavior:
- For data questions, you MUST call a tool.
- Prefer `run_bi_query` for most analytics requests.
- For viability/trajectory/health questions, prefer `assess_business_health`.
- Use legacy tools only when the user asks specifically for those exact outputs.
- Never fabricate numbers or metrics.
- If a requested field does not exist, still call `run_bi_query` and let the tool return warnings.

run_bi_query planning guidelines:
- dataset: `deals` unless the user asks about work orders.
- filters: convert explicit constraints (sector/status/owner/ranges).
- metrics: choose from count/sum/avg/min/max.
- metric.field MUST be a source column (e.g., deal_value, wo_value, status, owner), not an output alias.
- metric.alias is only the output label (e.g., projected_revenue).
- For derived intents (profit/margin), request those terms directly; the tool will derive them from available columns when possible.
- If ingredients are missing, still run best-effort; tool warnings will describe assumptions/proxy usage.
- For sector questions, always include a sector filter.
- If sector text appears umbrella/ambiguous (e.g., "energy"), prefer `operator: "in"` with likely related sectors.
- If exact sector may not exist, still proceed with best related sectors; tool warnings will indicate fallback used.
- group_by: add when user asks comparisons/rankings/breakdowns.
- time_scope: use quarter/year/last_year when explicitly implied, otherwise all.
- sort_by + sort_order + limit for top/bottom style questions.

""" + schema_context

    return client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        tools=tool_schemas,
        tool_choice="auto",
    )


def _force_run_bi_query_call(user_query: str, tool_schemas: list[dict[str, Any]]) -> Any:
    return client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": "The previous attempt did not call a tool. Call run_bi_query now with best-effort arguments.",
            },
            {"role": "user", "content": user_query},
        ],
        tools=tool_schemas,
        tool_choice={"type": "function", "function": {"name": "run_bi_query"}},
    )


def _render_health_response(result: dict[str, Any], user_query: str) -> str:
    if not isinstance(result, dict):
        return "Health assessment failed because the tool output was invalid."

    if "error" in result:
        return f"Health assessment failed: {result['error']}"

    score = result.get("health_score")
    flag = result.get("health_flag", "Unknown")
    trend = result.get("trend_signal", "Unknown")
    scope = str(result.get("time_scope_applied", infer_time_scope_from_query(user_query))).replace("_", " ")
    totals = result.get("totals") or {}

    lines = [
        f"Business health ({scope}): {flag} (score {score}/100).",
        f"Trend signal: {trend}.",
        (
            "Pipeline mix by value: "
            f"Won {totals.get('won_value', '$0.00')}, "
            f"Active {totals.get('active_value', '$0.00')}, "
            f"Risk {totals.get('risk_value', '$0.00')}."
        ),
        (
            "Deal counts: "
            f"Won {totals.get('won_deal_count', 0)}, "
            f"Active {totals.get('active_deal_count', 0)}, "
            f"Risk {totals.get('risk_deal_count', 0)}, "
            f"Total {totals.get('total_deal_count', 0)}."
        ),
    ]

    row_count = result.get("row_count_after_time_scope")
    if isinstance(row_count, int):
        lines.append(f"Rows considered after time filter: {row_count}.")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("Warnings/assumptions:")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def run_agent(user_query: str) -> str:
    log_trace("Sending query to LLM")

    tool_schemas = _build_tool_schemas()

    if detect_health_intent(user_query) and "assess_business_health" in TOOLS:
        log_trace("Intent override: routing to assess_business_health")
        health_args = {"time_scope": infer_time_scope_from_query(user_query)}
        log_trace(f"Tool arguments: {health_args}")
        try:
            health_result = TOOLS["assess_business_health"]["handler"](**health_args)
            log_trace("assess_business_health execution completed")
            return _render_health_response(health_result, user_query)
        except Exception as exc:
            log_trace(f"assess_business_health execution failed: {exc}")
            return f"Health assessment failed: {exc}"

    response = _call_llm_for_tool_selection(user_query, tool_schemas)
    message = response.choices[0].message

    if not message.tool_calls:
        log_trace("No tool selected on first pass. Forcing run_bi_query fallback.")
        forced_response = _force_run_bi_query_call(user_query, tool_schemas)
        message = forced_response.choices[0].message

    if message.tool_calls:
        tool_messages = []

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if tool_name not in TOOLS:
                result = {"error": f"Unknown tool '{tool_name}' selected by model."}
                log_trace(result["error"])
            else:
                arguments = _sanitize_tool_args(tool_name, arguments, user_query)
                log_trace(f"LLM selected tool: {tool_name}")
                log_trace(f"Tool arguments: {arguments}")

                try:
                    result = TOOLS[tool_name]["handler"](**arguments)
                    log_trace(f"{tool_name} execution completed")
                except Exception as exc:
                    result = {"error": str(exc)}
                    log_trace(f"{tool_name} execution failed: {exc}")

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        final_response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a BI analyst.

You MUST answer strictly from tool output.
Do NOT add numbers not present in tool output.
If output has warnings, explain them briefly.
If warnings indicate sector fallback was used, explicitly state that exact sector was absent and list related sectors used.
If warnings indicate metric derivation/proxy usage (profit/margin), clearly state the assumption before giving numbers.
If output has errors or empty data, say so clearly and suggest the next best query.
Provide concise executive insight.
""",
                },
                {"role": "user", "content": user_query},
                message,
                *tool_messages,
            ],
        )

        return final_response.choices[0].message.content

    log_trace("Tool execution fallback failed")
    return (
        "I could not execute a BI tool for this query. "
        "Please rephrase with dataset intent (deals/work orders), metric, and optional filters."
    )
