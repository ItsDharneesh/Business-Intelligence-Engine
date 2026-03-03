from __future__ import annotations

import re
from typing import Any


SUPPORTED_DATASETS = {"deals", "work_orders"}
SUPPORTED_TIME_SCOPES = {"all", "quarter", "year", "last_year"}
SUPPORTED_AGGS = {"count", "sum", "avg", "mean", "min", "max"}
SUPPORTED_OPERATORS = {
    "eq", "=", "is",
    "ne", "!=", "not",
    "contains", "like",
    "not_contains", "not like",
    "in",
    "gt", ">", "gte", ">=",
    "lt", "<", "lte", "<=",
    "is_null", "is_not_null",
}

DEALS_FIELDS = {
    "item_id", "item_name", "deal_value", "status", "close_date", "sector", "owner",
    "profit", "profit_proxy", "margin_pct",
}
WORK_ORDER_FIELDS = {
    "item_id", "item_name", "wo_value", "execution_status", "sector",
}

DATASET_FIELDS = {
    "deals": DEALS_FIELDS,
    "work_orders": WORK_ORDER_FIELDS,
}

DEALS_ALIASES = {
    "revenue": "deal_value",
    "pipeline": "deal_value",
    "value": "deal_value",
    "amount": "deal_value",
    "stage": "status",
    "industry": "sector",
    "sales rep": "owner",
    "salesperson": "owner",
    "profit margin": "margin_pct",
}

WORK_ORDERS_ALIASES = {
    "value": "wo_value",
    "amount": "wo_value",
    "industry": "sector",
    "status": "execution_status",
}

DATASET_ALIASES = {
    "deals": DEALS_ALIASES,
    "work_orders": WORK_ORDERS_ALIASES,
}

OPERATOR_ALIASES = {
    "equals": "eq",
    "equal": "eq",
    "not_equals": "ne",
    "not_equal": "ne",
    "does_not_equal": "ne",
    "greater_than": "gt",
    "greater_than_or_equal": "gte",
    "less_than": "lt",
    "less_than_or_equal": "lte",
}

HEALTH_INTENT_PATTERNS = [
    r"\bhealthy\b",
    r"\bhealth\b",
    r"\bfinancial health\b",
    r"\brunway\b",
    r"\bwrong direction\b",
    r"\bright direction\b",
    r"\bmake it through\b",
    r"\bviable\b",
]


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_time_scope_from_query(user_query: str) -> str:
    q = (user_query or "").lower()
    if re.search(r"\blast year\b|\bprevious year\b|\bprior year\b", q):
        return "last_year"
    if re.search(r"\bquarter\b|\bq[1-4]\b", q):
        return "quarter"
    if re.search(r"\byear\b|\bannual\b|\bytd\b", q):
        return "year"
    return "all"


def detect_health_intent(user_query: str) -> bool:
    q = (user_query or "").lower()
    return any(re.search(pattern, q) for pattern in HEALTH_INTENT_PATTERNS)


def normalize_dataset(value: Any) -> str:
    normalized = _normalize_text(value).replace("-", "_")
    if normalized in {"workorders", "work_order", "wo", "workorders_board"}:
        return "work_orders"
    if normalized in SUPPORTED_DATASETS:
        return normalized
    return "deals"


def normalize_field(dataset: str, field: Any) -> str:
    raw = str(field or "").strip()
    if not raw:
        return ""

    dataset_key = normalize_dataset(dataset)
    known_fields = DATASET_FIELDS[dataset_key]
    if raw in known_fields:
        return raw

    normalized = _normalize_text(raw)
    if normalized in known_fields:
        return normalized

    alias_map = DATASET_ALIASES[dataset_key]
    mapped = alias_map.get(normalized)
    if mapped:
        return mapped

    return raw


def normalize_operator(value: Any) -> str:
    op = _normalize_text(value or "eq")
    op = OPERATOR_ALIASES.get(op, op)
    return op if op in SUPPORTED_OPERATORS else "eq"


def normalize_agg(value: Any) -> str:
    agg = _normalize_text(value or "count")
    if agg == "mean":
        agg = "avg"
    return agg if agg in SUPPORTED_AGGS else "count"


def constrain_run_bi_query_arguments(arguments: dict[str, Any], user_query: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    args = dict(arguments or {})

    dataset = normalize_dataset(args.get("dataset", "deals"))
    args["dataset"] = dataset

    time_scope = _normalize_text(args.get("time_scope") or "")
    if not time_scope:
        time_scope = infer_time_scope_from_query(user_query)
    if time_scope not in SUPPORTED_TIME_SCOPES:
        warnings.append(f"Unsupported time_scope '{time_scope}' was reset to all.")
        time_scope = "all"
    args["time_scope"] = time_scope

    sort_order = _normalize_text(args.get("sort_order") or "desc")
    args["sort_order"] = "asc" if sort_order == "asc" else "desc"

    limit = args.get("limit", 20)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        warnings.append("Invalid limit was reset to 20.")
        limit = 20
    args["limit"] = max(1, min(limit, 200))

    args["sort_by"] = normalize_field(dataset, args.get("sort_by", ""))

    group_by = args.get("group_by")
    if not isinstance(group_by, list):
        group_by = []
    args["group_by"] = [normalize_field(dataset, field) for field in group_by if str(field or "").strip()]

    filters = args.get("filters")
    if not isinstance(filters, list):
        filters = []
    constrained_filters: list[dict[str, Any]] = []
    for raw_filter in filters:
        if not isinstance(raw_filter, dict):
            continue
        constrained_filters.append(
            {
                "field": normalize_field(dataset, raw_filter.get("field", "")),
                "operator": normalize_operator(raw_filter.get("operator") or raw_filter.get("op") or "eq"),
                "value": raw_filter.get("value"),
            }
        )
    args["filters"] = constrained_filters

    metrics = args.get("metrics")
    if not isinstance(metrics, list):
        metrics = []
    constrained_metrics: list[dict[str, Any]] = []
    for raw_metric in metrics:
        if not isinstance(raw_metric, dict):
            continue
        metric = {
            "agg": normalize_agg(raw_metric.get("agg") or raw_metric.get("aggregation") or "count"),
            "field": normalize_field(dataset, raw_metric.get("field", "*")) if str(raw_metric.get("field", "*")).strip() != "*" else "*",
        }
        alias = str(raw_metric.get("alias", "") or "").strip()
        if alias:
            metric["alias"] = alias
        constrained_metrics.append(metric)
    args["metrics"] = constrained_metrics

    return args, warnings


def get_planner_schema_context() -> str:
    return """
Schema Catalog (enforced by runtime sanitizer):
- datasets:
  - deals fields: item_id, item_name, deal_value, status, close_date, sector, owner, profit, profit_proxy, margin_pct
  - work_orders fields: item_id, item_name, wo_value, execution_status, sector
- operators: eq/ne/contains/not_contains/in/gt/gte/lt/lte/is_null/is_not_null
- aggs: count/sum/avg/min/max
- time_scope: all/quarter/year/last_year
- for health viability questions, prefer `assess_business_health` instead of free-form run_bi_query.
"""
