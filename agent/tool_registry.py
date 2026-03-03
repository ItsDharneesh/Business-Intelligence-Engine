from __future__ import annotations

import re
from typing import Any

import pandas as pd

from config import DEALS_BOARD_ID, WORK_ORDERS_BOARD_ID
from services.bi_engine import (
    average_deal_size_per_owner,
    calculate_pipeline_value,
    conversion_rate,
    delayed_work_order_revenue,
    execute_bi_query,
    filter_by_sector,
    filter_closed_deals,
    filter_current_quarter,
    filter_current_year,
    format_currency,
    owner_total_performance,
    revenue_by_sector,
)
from services.monday_client import fetch_board_items
from services.normalization import normalize_items
from utils.trace_logger import log_trace


DEALS_FIELD_ALIASES = {
    "revenue": "deal_value",
    "pipeline": "deal_value",
    "pipeline value": "deal_value",
    "deal value": "deal_value",
    "value": "deal_value",
    "amount": "deal_value",
    "profit": "profit",
    "margin": "margin_pct",
    "profit margin": "margin_pct",
    "status": "status",
    "stage": "status",
    "sector": "sector",
    "industry": "sector",
    "owner": "owner",
    "sales rep": "owner",
    "salesperson": "owner",
    "close date": "close_date",
    "date": "close_date",
    "deal": "item_name",
    "deal name": "item_name",
    "id": "item_id",
}

WORK_ORDERS_FIELD_ALIASES = {
    "work order value": "wo_value",
    "wo value": "wo_value",
    "value": "wo_value",
    "amount": "wo_value",
    "sector": "sector",
    "industry": "sector",
    "execution status": "execution_status",
    "status": "execution_status",
    "work order": "item_name",
    "work order name": "item_name",
    "id": "item_id",
}

DATASET_CONFIG = {
    "deals": {
        "board_id": DEALS_BOARD_ID,
        "aliases": DEALS_FIELD_ALIASES,
        "date_columns": ["close_date"],
    },
    "work_orders": {
        "board_id": WORK_ORDERS_BOARD_ID,
        "aliases": WORK_ORDERS_FIELD_ALIASES,
        "date_columns": ["execution_date", "close_date", "date"],
    },
}


def _normalize_term(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    available = set(df.columns)
    for col in candidates:
        if col in available:
            return col
    return None


def _augment_derived_columns(df: pd.DataFrame, dataset_key: str) -> tuple[pd.DataFrame, list[str]]:
    working = df.copy()
    warnings: list[str] = []

    revenue_candidates = ["deal_value", "revenue", "wo_value", "amount", "value"]
    cost_candidates = ["cost", "total_cost", "project_cost", "expense", "expenses", "wo_cost"]

    revenue_col = _first_existing_column(working, revenue_candidates)
    cost_col = _first_existing_column(working, cost_candidates)

    if "profit" not in working.columns and revenue_col and cost_col:
        working["profit"] = _to_numeric_series(working[revenue_col]) - _to_numeric_series(working[cost_col])
        warnings.append(
            f"Derived 'profit' as {revenue_col} - {cost_col}."
        )

    if "profit" not in working.columns and "profit_proxy" not in working.columns and revenue_col and not cost_col:
        working["profit_proxy"] = _to_numeric_series(working[revenue_col])
        warnings.append(
            f"No cost column found; derived 'profit_proxy' from {revenue_col} as a revenue-based proxy."
        )

    profit_base = "profit" if "profit" in working.columns else ("profit_proxy" if "profit_proxy" in working.columns else None)
    if "margin_pct" not in working.columns and revenue_col and profit_base:
        revenue = _to_numeric_series(working[revenue_col]).replace(0, pd.NA)
        profit = _to_numeric_series(working[profit_base])
        working["margin_pct"] = ((profit / revenue) * 100).fillna(0)
        if profit_base == "profit":
            warnings.append("Derived 'margin_pct' as profit/revenue * 100.")
        else:
            warnings.append("Derived 'margin_pct' from profit proxy; treat as approximate.")

    return working, warnings


def _normalize_dataset_name(dataset: str) -> str:
    value = str(dataset or "deals").strip().lower().replace("-", "_")
    if value in {"workorders", "work_order", "wo", "workorders_board"}:
        return "work_orders"
    return value


def _load_dataset_df(dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_key = _normalize_dataset_name(dataset)
    config = DATASET_CONFIG.get(dataset_key)

    if not config:
        return pd.DataFrame(), {
            "error": f"Unsupported dataset '{dataset}'. Use one of: {', '.join(DATASET_CONFIG.keys())}."
        }

    board_id = config.get("board_id")
    if not board_id:
        return pd.DataFrame(), {"error": f"Board ID is missing for dataset '{dataset_key}'."}

    raw = fetch_board_items(int(board_id))
    df = normalize_items(raw)
    if df.empty:
        return df, {"error": f"Dataset '{dataset_key}' returned no data."}

    return df, {"dataset": dataset_key}


def _append_currency_formats(result: dict[str, Any]) -> None:
    rows = result.get("result") or []
    metrics = result.get("metrics_applied") or []
    aliases = {m.get("alias") for m in metrics if isinstance(m, dict)}

    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in list(row.items()):
            if key not in aliases:
                continue
            if not isinstance(value, (int, float)):
                continue
            normalized_key = key.lower()
            if "value" in normalized_key or normalized_key.endswith("_sum") or normalized_key.endswith("_avg"):
                row[f"{key}_formatted"] = format_currency(float(value))


def _bucket_status(status: Any) -> str:
    text = _normalize_term(status)
    if not text:
        return "unknown"

    won_tokens = {"won", "closed won", "closed"}
    active_tokens = {"open", "negotiation", "contract", "proposal", "qualified", "pipeline"}
    risk_tokens = {"dead", "lost", "on hold", "hold", "cancelled", "canceled"}

    if any(token in text for token in won_tokens):
        return "won"
    if any(token in text for token in active_tokens):
        return "active"
    if any(token in text for token in risk_tokens):
        return "risk"
    return "unknown"


def compute_health_from_status_rows(
    status_rows: list[dict[str, Any]],
    *,
    time_scope_applied: str,
) -> dict[str, Any]:
    if not status_rows:
        return {
            "time_scope_applied": time_scope_applied,
            "health_score": 0,
            "health_flag": "Insufficient data",
            "trend_signal": "No data",
            "status_breakdown": [],
            "warnings": ["No status rows were available for health assessment."],
        }

    totals = {
        "won_value": 0.0,
        "active_value": 0.0,
        "risk_value": 0.0,
        "unknown_value": 0.0,
        "won_count": 0,
        "active_count": 0,
        "risk_count": 0,
        "unknown_count": 0,
    }
    breakdown: list[dict[str, Any]] = []

    for row in status_rows:
        status = row.get("status")
        value = float(row.get("status_value", 0) or 0)
        count = int(row.get("deal_count", 0) or 0)
        bucket = _bucket_status(status)

        totals[f"{bucket}_value"] += value
        totals[f"{bucket}_count"] += count

        breakdown.append(
            {
                "status": status if status not in {None, ""} else "(blank)",
                "bucket": bucket,
                "deal_count": count,
                "status_value_raw": value,
                "status_value": format_currency(value),
            }
        )

    total_value = (
        totals["won_value"]
        + totals["active_value"]
        + totals["risk_value"]
        + totals["unknown_value"]
    )
    total_count = (
        totals["won_count"]
        + totals["active_count"]
        + totals["risk_count"]
        + totals["unknown_count"]
    )

    if total_value <= 0 and total_count <= 0:
        return {
            "time_scope_applied": time_scope_applied,
            "health_score": 0,
            "health_flag": "Insufficient data",
            "trend_signal": "No data",
            "status_breakdown": breakdown,
            "warnings": ["Rows exist but both value and counts are zero."],
        }

    won_value_ratio = (totals["won_value"] / total_value) if total_value > 0 else 0.0
    active_value_ratio = (totals["active_value"] / total_value) if total_value > 0 else 0.0
    risk_value_ratio = (totals["risk_value"] / total_value) if total_value > 0 else 0.0
    won_count_ratio = (totals["won_count"] / total_count) if total_count > 0 else 0.0
    risk_count_ratio = (totals["risk_count"] / total_count) if total_count > 0 else 0.0

    score = (
        50
        + 35 * won_value_ratio
        + 15 * active_value_ratio
        - 45 * risk_value_ratio
        + 10 * won_count_ratio
        - 5 * risk_count_ratio
    )
    score = int(max(0, min(round(score), 100)))

    if score >= 70:
        health_flag = "Healthy"
    elif score >= 50:
        health_flag = "Watch"
    else:
        health_flag = "At Risk"

    if risk_value_ratio > won_value_ratio:
        trend_signal = "Downward pressure"
    elif won_value_ratio >= risk_value_ratio and active_value_ratio >= 0.2:
        trend_signal = "Positive momentum"
    else:
        trend_signal = "Mixed"

    warnings: list[str] = []
    if totals["unknown_count"] > 0:
        warnings.append("Some statuses were unclassified and counted under 'unknown'.")

    return {
        "time_scope_applied": time_scope_applied,
        "health_score": score,
        "health_flag": health_flag,
        "trend_signal": trend_signal,
        "totals": {
            "pipeline_value_raw": total_value,
            "pipeline_value": format_currency(total_value),
            "won_value_raw": totals["won_value"],
            "won_value": format_currency(totals["won_value"]),
            "active_value_raw": totals["active_value"],
            "active_value": format_currency(totals["active_value"]),
            "risk_value_raw": totals["risk_value"],
            "risk_value": format_currency(totals["risk_value"]),
            "total_deal_count": total_count,
            "won_deal_count": totals["won_count"],
            "active_deal_count": totals["active_count"],
            "risk_deal_count": totals["risk_count"],
            "unknown_deal_count": totals["unknown_count"],
        },
        "ratios": {
            "won_value_ratio": round(won_value_ratio, 4),
            "active_value_ratio": round(active_value_ratio, 4),
            "risk_value_ratio": round(risk_value_ratio, 4),
            "won_count_ratio": round(won_count_ratio, 4),
            "risk_count_ratio": round(risk_count_ratio, 4),
        },
        "status_breakdown": sorted(
            breakdown,
            key=lambda item: float(item.get("status_value_raw", 0) or 0),
            reverse=True,
        ),
        "warnings": warnings,
    }


def assess_business_health_tool(time_scope: str = "quarter") -> dict[str, Any]:
    df, metadata = _load_dataset_df("deals")
    if "error" in metadata:
        return metadata

    result = execute_bi_query(
        df,
        filters=[],
        metrics=[
            {"field": "deal_value", "agg": "sum", "alias": "status_value"},
            {"field": "*", "agg": "count", "alias": "deal_count"},
        ],
        group_by=["status"],
        time_scope=time_scope,
        sort_by="status_value",
        sort_order="desc",
        limit=200,
        field_aliases=DEALS_FIELD_ALIASES,
        date_columns=DATASET_CONFIG["deals"]["date_columns"],
    )

    health = compute_health_from_status_rows(
        status_rows=result.get("result", []),
        time_scope_applied=result.get("time_scope_applied", "all"),
    )

    merged_warnings = [*(result.get("warnings") or []), *(health.get("warnings") or [])]
    if merged_warnings:
        health["warnings"] = merged_warnings

    health["row_count_after_time_scope"] = result.get("row_count_after_time_scope", 0)
    return health


def _repair_metrics(
    metrics: list[dict[str, Any]] | None,
    dataset_key: str,
    available_columns: list[str],
    aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not metrics:
        return [], []

    warnings: list[str] = []
    repaired: list[dict[str, Any]] = []

    default_value_field = "deal_value" if dataset_key == "deals" else "wo_value"
    if default_value_field not in available_columns:
        value_candidates = [c for c in available_columns if c.endswith("_value")]
        default_value_field = value_candidates[0] if value_candidates else default_value_field

    profit_terms = {"profit", "net profit", "gross profit", "profitability"}
    margin_terms = {"margin", "profit margin", "margin percent", "margin_pct"}

    def _looks_like_profit(text: str) -> bool:
        return text in profit_terms or text.endswith("_profit") or text.endswith("profit")

    def _looks_like_margin(text: str) -> bool:
        return text in margin_terms or text.endswith("_margin") or "margin" in text

    for metric in metrics:
        if not isinstance(metric, dict):
            repaired.append(metric)
            continue

        item = dict(metric)
        raw_field = str(item.get("field", "") or "").strip()
        raw_field_norm = _normalize_term(raw_field)
        raw_alias = str(item.get("alias", "") or "").strip()
        raw_alias_norm = _normalize_term(raw_alias)
        agg = str(item.get("agg", "") or item.get("aggregation", "") or "").strip().lower()

        if not raw_field:
            if _looks_like_profit(raw_alias_norm):
                if "profit" in available_columns:
                    item["field"] = "profit"
                    warnings.append("Mapped profit metric alias to derived field 'profit'.")
                elif "profit_proxy" in available_columns:
                    item["field"] = "profit_proxy"
                    warnings.append("Mapped profit metric alias to 'profit_proxy' (revenue-based approximation).")
                else:
                    item["field"] = default_value_field
                    warnings.append(
                        f"Profit metric alias mapped to '{default_value_field}' because profit ingredients were unavailable."
                    )
                repaired.append(item)
                continue

            if _looks_like_margin(raw_alias_norm):
                if "margin_pct" in available_columns:
                    item["field"] = "margin_pct"
                    if not item.get("agg"):
                        item["agg"] = "avg"
                    warnings.append("Mapped margin metric alias to derived field 'margin_pct'.")
                    repaired.append(item)
                    continue

            if agg in {"sum", "avg", "mean", "min", "max"}:
                item["field"] = default_value_field
                warnings.append(
                    f"Metric without field was mapped to '{default_value_field}' for aggregation '{agg}'."
                )
            repaired.append(item)
            continue

        if raw_field in available_columns:
            repaired.append(item)
            continue

        alias_field = aliases.get(raw_field_norm)
        if alias_field and alias_field in available_columns:
            item["field"] = alias_field
            warnings.append(f"Metric field '{raw_field}' mapped to '{alias_field}'.")
            repaired.append(item)
            continue

        if _looks_like_profit(raw_field_norm) or _looks_like_profit(raw_alias_norm):
            if "profit" in available_columns:
                item["field"] = "profit"
                warnings.append(f"Metric field '{raw_field}' mapped to derived 'profit'.")
            elif "profit_proxy" in available_columns:
                item["field"] = "profit_proxy"
                warnings.append(
                    f"Metric field '{raw_field}' mapped to 'profit_proxy' (revenue-based approximation)."
                )
            else:
                item["field"] = default_value_field
                warnings.append(
                    f"Metric field '{raw_field}' mapped to '{default_value_field}' because no cost/profit fields were available."
                )
            repaired.append(item)
            continue

        if _looks_like_margin(raw_field_norm) or _looks_like_margin(raw_alias_norm):
            if "margin_pct" in available_columns:
                item["field"] = "margin_pct"
                if agg not in {"avg", "mean", "min", "max"}:
                    item["agg"] = "avg"
                warnings.append(f"Metric field '{raw_field}' mapped to derived 'margin_pct'.")
                repaired.append(item)
                continue

        if raw_alias_norm and raw_alias_norm in {"projected_revenue", "revenue", "pipeline_value", "deal_value", "wo_value"}:
            item["field"] = default_value_field
            warnings.append(
                f"Metric field '{raw_field}' repaired to '{default_value_field}' using alias context '{raw_alias_norm}'."
            )
            repaired.append(item)
            continue

        if agg in {"sum", "avg", "mean", "min", "max"} and raw_field_norm.endswith("_revenue"):
            item["field"] = default_value_field
            warnings.append(
                f"Metric field '{raw_field}' repaired to '{default_value_field}' for numeric aggregation '{agg}'."
            )
            repaired.append(item)
            continue

        repaired.append(item)

    return repaired, warnings


# ==========================================================
# GENERIC PRODUCTION TOOL
# ==========================================================


def run_bi_query_tool(
    dataset: str = "deals",
    filters: list[dict[str, Any]] | None = None,
    metrics: list[dict[str, Any]] | None = None,
    group_by: list[str] | None = None,
    time_scope: str = "all",
    sort_by: str = "",
    sort_order: str = "desc",
    limit: int = 20,
) -> dict[str, Any]:
    dataset_key = _normalize_dataset_name(dataset)
    log_trace(f"run_bi_query called for dataset={dataset_key}")

    df, metadata = _load_dataset_df(dataset_key)
    if "error" in metadata:
        return metadata

    df, derivation_warnings = _augment_derived_columns(df, dataset_key)

    config = DATASET_CONFIG[dataset_key]
    repaired_metrics, repair_warnings = _repair_metrics(
        metrics=metrics,
        dataset_key=dataset_key,
        available_columns=list(df.columns),
        aliases=config["aliases"],
    )
    result = execute_bi_query(
        df,
        filters=filters,
        metrics=repaired_metrics or metrics,
        group_by=group_by,
        time_scope=time_scope,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        field_aliases=config["aliases"],
        date_columns=config["date_columns"],
    )

    _append_currency_formats(result)
    merged_warnings = [*derivation_warnings, *repair_warnings, *(result.get("warnings") or [])]
    if merged_warnings:
        result["warnings"] = merged_warnings

    return {
        "dataset": dataset_key,
        **result,
    }


# ==========================================================
# LEGACY TOOLS (BACKWARD COMPATIBILITY)
# ==========================================================


def fetch_deals_tool(sector: str = "", time_scope: str = "all") -> dict[str, Any]:
    log_trace("Fetching Deals board (Live API)")
    raw = fetch_board_items(int(DEALS_BOARD_ID))

    boards = raw.get("data", {}).get("boards", [])
    log_trace(f"Boards returned: {len(boards)}")

    if boards:
        items = boards[0].get("items_page", {}).get("items", [])
        log_trace(f"Items fetched from board: {len(items)}")

    df = normalize_items(raw)
    log_trace(f"After normalization: {df.shape}")

    if df.empty:
        log_trace("DataFrame empty after normalization")
        return {"error": "Deals board returned no data."}

    base_df = df.copy()

    sector_matched_count = None
    if sector:
        df = filter_by_sector(df, sector)
        sector_matched_count = int(len(df))
        log_trace(f"After sector filter: {df.shape}")

    normalized_scope = str(time_scope or "all").strip().lower()
    if normalized_scope in {"quarter", "qtr", "q"}:
        df = filter_current_quarter(df)
        applied_scope = "quarter"
    elif normalized_scope in {"year", "annual", "ytd", "current_year"}:
        df = filter_current_year(df)
        applied_scope = "year"
    else:
        applied_scope = "all"

    pipeline_value = calculate_pipeline_value(df)
    result = {
        "pipeline_value_raw": pipeline_value,
        "pipeline_value": format_currency(pipeline_value),
        "deal_count": int(len(df)),
        "time_scope_applied": applied_scope,
    }

    if sector:
        result["sector_filter"] = sector
        result["sector_match_count_before_time_filter"] = int(sector_matched_count or 0)

        if sector_matched_count == 0 and "sector" in base_df.columns:
            available = (
                base_df["sector"]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            result["available_sectors"] = sorted(available)
            result["note"] = "No deals matched the requested sector text."
        elif sector_matched_count and len(df) == 0:
            if applied_scope == "quarter":
                result["note"] = "Sector matched deals, but none are in the current quarter."
            elif applied_scope == "year":
                result["note"] = "Sector matched deals, but none are in the current year."
            else:
                result["note"] = "Sector matched deals, but final result is empty after filters."

    return result


def sector_performance_tool() -> dict[str, Any]:
    raw = fetch_board_items(int(DEALS_BOARD_ID))
    df = normalize_items(raw)

    df = filter_closed_deals(df)
    df = filter_current_quarter(df)

    sector_revenue = revenue_by_sector(df)

    if not sector_revenue:
        return {
            "top_sector": None,
            "sector_revenue": {},
            "total_deals": 0,
        }

    top_sector = max(sector_revenue, key=sector_revenue.get)

    return {
        "top_sector": top_sector,
        "top_sector_revenue": sector_revenue[top_sector],
        "sector_revenue": sector_revenue,
        "total_deals": int(len(df)),
    }


def fetch_work_orders_tool(sector: str = "") -> dict[str, Any]:
    raw = fetch_board_items(int(WORK_ORDERS_BOARD_ID))
    df = normalize_items(raw)

    if sector:
        df = filter_by_sector(df, sector)

    return {
        "work_order_count": int(len(df)),
    }


def cross_board_pipeline_health(sector: str = "") -> dict[str, Any]:
    deals_result = fetch_deals_tool(sector)
    work_orders_result = fetch_work_orders_tool(sector)

    deal_count = deals_result.get("deal_count", 0)
    work_order_count = work_orders_result.get("work_order_count", 0)

    if deal_count == 0:
        health_flag = "No active deals"
    elif work_order_count > deal_count:
        health_flag = "Potential operational strain"
    elif work_order_count == deal_count:
        health_flag = "Balanced"
    else:
        health_flag = "Strong pipeline support"

    return {
        "pipeline_value": deals_result.get("pipeline_value"),
        "deal_count": deal_count,
        "work_order_count": work_order_count,
        "health_flag": health_flag,
    }


def average_deal_size_tool() -> dict[str, Any]:
    raw = fetch_board_items(int(DEALS_BOARD_ID))
    df = normalize_items(raw)

    df = filter_closed_deals(df)
    df = filter_current_quarter(df)

    result = average_deal_size_per_owner(df)
    return {"average_deal_size_per_owner": result}


def owner_performance_tool() -> dict[str, Any]:
    raw = fetch_board_items(int(DEALS_BOARD_ID))
    df = normalize_items(raw)

    df = filter_closed_deals(df)
    df = filter_current_quarter(df)

    result = owner_total_performance(df)
    return {"owner_total_revenue": result}


def conversion_rate_tool() -> dict[str, Any]:
    deals_raw = fetch_board_items(int(DEALS_BOARD_ID))
    deals_df = normalize_items(deals_raw)
    deals_df = filter_closed_deals(deals_df)
    deals_df = filter_current_quarter(deals_df)

    work_raw = fetch_board_items(int(WORK_ORDERS_BOARD_ID))
    work_df = normalize_items(work_raw)

    rate = conversion_rate(len(deals_df), len(work_df))

    return {
        "closed_deals": len(deals_df),
        "work_orders": len(work_df),
        "conversion_percentage": rate,
    }


def delayed_revenue_tool() -> dict[str, Any]:
    raw = fetch_board_items(int(WORK_ORDERS_BOARD_ID))
    df = normalize_items(raw)

    revenue = delayed_work_order_revenue(df)

    return {
        "delayed_revenue_raw": revenue,
        "delayed_revenue": format_currency(revenue),
    }


# ==========================================================
# TOOL REGISTRY
# ==========================================================


TOOLS = {
    "assess_business_health": {
        "description": (
            "Deterministic health assessment for the business pipeline in a time period. "
            "Use this for viability/trajectory questions like 'Are we healthy this quarter?'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "time_scope": {
                    "type": "string",
                    "description": "all, quarter, year, or last_year",
                }
            },
        },
        "handler": assess_business_health_tool,
    },
    "run_bi_query": {
        "description": (
            "Universal BI query tool. Use for most questions. "
            "Supports filters, metrics, group_by, time_scope, sorting, and limits "
            "for both deals and work_orders datasets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset": {
                    "type": "string",
                    "description": "deals or work_orders",
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "operator": {"type": "string"},
                            "value": {},
                        },
                        "required": ["field", "operator"],
                    },
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "agg": {"type": "string"},
                            "alias": {"type": "string"},
                        },
                        "required": ["agg"],
                    },
                },
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "time_scope": {
                    "type": "string",
                    "description": "all, quarter, year, or last_year",
                },
                "sort_by": {"type": "string"},
                "sort_order": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        "handler": run_bi_query_tool,
    },
    "fetch_deals": {
        "description": "Legacy tool: fetch total pipeline value and deal count.",
        "parameters": {
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
                "time_scope": {"type": "string"},
            },
        },
        "handler": fetch_deals_tool,
    },
    "sector_performance": {
        "description": "Legacy tool: top performing sector this quarter based on closed deal value.",
        "parameters": {"type": "object", "properties": {}},
        "handler": sector_performance_tool,
    },
    "cross_board_health": {
        "description": "Legacy tool: pipeline and workload comparison.",
        "parameters": {
            "type": "object",
            "properties": {
                "sector": {"type": "string"},
            },
        },
        "handler": cross_board_pipeline_health,
    },
    "average_deal_size": {
        "description": "Legacy tool: average closed deal size per owner for current quarter.",
        "parameters": {"type": "object", "properties": {}},
        "handler": average_deal_size_tool,
    },
    "owner_performance": {
        "description": "Legacy tool: owner ranking by closed deal value for current quarter.",
        "parameters": {"type": "object", "properties": {}},
        "handler": owner_performance_tool,
    },
    "conversion_rate": {
        "description": "Legacy tool: percentage of closed deals converted into work orders.",
        "parameters": {"type": "object", "properties": {}},
        "handler": conversion_rate_tool,
    },
    "delayed_revenue": {
        "description": "Legacy tool: revenue impact of delayed work orders.",
        "parameters": {"type": "object", "properties": {}},
        "handler": delayed_revenue_tool,
    },
}
