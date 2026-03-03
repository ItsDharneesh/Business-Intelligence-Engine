from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


# ==========================================================
# TEXT NORMALIZATION HELPERS
# ==========================================================

SECTOR_RELATED_TERMS: dict[str, list[str]] = {
    "energy": ["renewables", "powerline", "power line"],
    "infra": ["infrastructure", "construction", "railways"],
    "infrastructure": ["construction", "railways"],
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(value: Any) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()
    return {token for token in normalized.split(" ") if token}


def _apply_sector_filter_with_related(df: pd.DataFrame, requested_value: str) -> tuple[pd.DataFrame, str | None]:
    if df.empty or "sector" not in df.columns:
        return df, None

    requested_norm = _normalize_text(requested_value)
    if not requested_norm:
        return df, None

    available_raw = (
        df["sector"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )
    available_norm = {_normalize_text(value) for value in available_raw}

    exact_exists = requested_norm in available_norm
    candidate_terms = [requested_norm] if exact_exists else [requested_norm, *SECTOR_RELATED_TERMS.get(requested_norm, [])]

    matched_frames: list[pd.DataFrame] = []
    for term in dict.fromkeys(candidate_terms):
        term_df = filter_by_sector(df, term)
        if not term_df.empty:
            matched_frames.append(term_df)

    if not matched_frames:
        return df.iloc[0:0], None

    combined = pd.concat(matched_frames, ignore_index=False).drop_duplicates()

    if exact_exists:
        return combined, None

    matched_sectors = (
        combined["sector"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    if matched_sectors:
        note = (
            f"No exact sector '{requested_value}' found. "
            f"Used related sectors: {', '.join(sorted(map(str, matched_sectors)))}."
        )
        return combined, note

    return combined, None


# ==========================================================
# FILTER BY SECTOR
# ==========================================================


def filter_by_sector(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    if df.empty or not sector or "sector" not in df.columns:
        return df

    def _normalize_sector_query(value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            return normalized

        stopwords = {
            "for", "the", "a", "an", "in", "of", "our", "this", "that",
            "current", "quarter", "sector", "industry", "vertical", "domain",
            "pipeline", "looking", "how", "is", "are", "we", "currently",
        }
        tokens = [token for token in normalized.split(" ") if token and token not in stopwords]
        return " ".join(tokens).strip()

    query_norm = _normalize_sector_query(sector) or _normalize_text(sector)
    query_tokens = _tokenize(query_norm)
    if not query_norm:
        return df

    sector_aliases = {
        "mining": "mining",
        "energy": "powerline",
        "renewables": "renewables",
        "others": "others",
        "railways": "railways",
        "construction": "construction",
    }
    expanded_query = _normalize_text(sector_aliases.get(query_norm, query_norm))
    expanded_tokens = _tokenize(expanded_query)

    def _is_match(cell_value: str) -> bool:
        cell_norm = _normalize_text(cell_value)
        if not cell_norm:
            return False

        if (
            query_norm in cell_norm
            or cell_norm in query_norm
            or expanded_query in cell_norm
            or cell_norm in expanded_query
        ):
            return True

        cell_tokens = _tokenize(cell_norm)
        overlap = query_tokens.intersection(cell_tokens) or expanded_tokens.intersection(cell_tokens)
        if overlap and max(len(query_tokens), 1) > 0:
            overlap_ratio = len(overlap) / max(min(len(query_tokens), 2), min(len(expanded_tokens), 2), 1)
            if overlap_ratio >= 0.5:
                return True

        similarity = SequenceMatcher(None, expanded_query, cell_norm).ratio()
        return similarity >= 0.75

    return df[df["sector"].astype(str).apply(_is_match)]


# ==========================================================
# FILTER CLOSED DEALS
# ==========================================================


def filter_closed_deals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return df

    return df[
        df["status"]
        .astype(str)
        .str.lower()
        .str.contains(r"\b(won|closed)\b", regex=True, na=False)
    ]


# ==========================================================
# FILTER CURRENT QUARTER / YEAR
# ==========================================================


def filter_current_quarter(df: pd.DataFrame, date_column: str = "close_date") -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df

    temp_df = df.copy()
    temp_df[date_column] = pd.to_datetime(temp_df[date_column], errors="coerce")

    now = datetime.now()
    current_year = now.year
    current_quarter = (now.month - 1) // 3 + 1

    return temp_df[
        (temp_df[date_column].dt.year == current_year)
        & (temp_df[date_column].dt.quarter == current_quarter)
    ]


def filter_current_year(df: pd.DataFrame, date_column: str = "close_date") -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df

    temp_df = df.copy()
    temp_df[date_column] = pd.to_datetime(temp_df[date_column], errors="coerce")
    current_year = datetime.now().year

    return temp_df[temp_df[date_column].dt.year == current_year]


def filter_last_year(df: pd.DataFrame, date_column: str = "close_date") -> pd.DataFrame:
    if df.empty or date_column not in df.columns:
        return df

    temp_df = df.copy()
    temp_df[date_column] = pd.to_datetime(temp_df[date_column], errors="coerce")
    last_year = datetime.now().year - 1
    return temp_df[temp_df[date_column].dt.year == last_year]


# ==========================================================
# SAFE CURRENCY / NUMERIC HELPERS
# ==========================================================


def _clean_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def _coerce_numeric_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field not in df.columns:
        return pd.Series([0] * len(df), index=df.index, dtype="float64")

    series = df[field]
    normalized_field = _normalize_text(field)
    if "value" in normalized_field or normalized_field.endswith("amount"):
        return _clean_currency(series)
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ==========================================================
# LEGACY METRICS
# ==========================================================


def calculate_pipeline_value(df: pd.DataFrame) -> float:
    if df.empty or "deal_value" not in df.columns:
        return 0.0
    return float(_clean_currency(df["deal_value"]).sum())


def revenue_by_sector(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "sector" not in df.columns or "deal_value" not in df.columns:
        return {}

    temp_df = df.copy()
    temp_df["deal_value"] = _clean_currency(temp_df["deal_value"])
    grouped = temp_df.groupby("sector", dropna=True)["deal_value"].sum()
    return grouped.sort_values(ascending=False).to_dict()


def average_deal_size_per_owner(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "owner" not in df.columns or "deal_value" not in df.columns:
        return {}

    temp_df = df.copy()
    temp_df["deal_value"] = _clean_currency(temp_df["deal_value"])
    grouped = temp_df.groupby("owner", dropna=True)["deal_value"].mean()
    return grouped.sort_values(ascending=False).to_dict()


def owner_total_performance(df: pd.DataFrame) -> dict[str, float]:
    if df.empty or "owner" not in df.columns or "deal_value" not in df.columns:
        return {}

    temp_df = df.copy()
    temp_df["deal_value"] = _clean_currency(temp_df["deal_value"])
    grouped = temp_df.groupby("owner", dropna=True)["deal_value"].sum()
    return grouped.sort_values(ascending=False).to_dict()


def conversion_rate(closed_deals_count: int, work_order_count: int) -> float:
    if closed_deals_count == 0:
        return 0.0
    return round((work_order_count / closed_deals_count) * 100, 2)


def delayed_work_order_revenue(df: pd.DataFrame) -> float:
    if df.empty or "execution_status" not in df.columns or "wo_value" not in df.columns:
        return 0.0

    delayed = df[
        df["execution_status"]
        .astype(str)
        .str.lower()
        .str.contains("delay", na=False)
    ].copy()

    delayed["wo_value"] = _clean_currency(delayed["wo_value"])
    return float(delayed["wo_value"].sum())


# ==========================================================
# GENERIC QUERY PRIMITIVES (PRODUCTION ROUTING)
# ==========================================================


def resolve_field_name(field: str, available_columns: list[str], field_aliases: dict[str, str] | None = None) -> str | None:
    if not field:
        return None

    aliases = field_aliases or {}
    normalized_lookup = {_normalize_text(col): col for col in available_columns}

    if field in available_columns:
        return field

    normalized = _normalize_text(field)

    if normalized in normalized_lookup:
        return normalized_lookup[normalized]

    canonical = aliases.get(normalized)
    if canonical and canonical in available_columns:
        return canonical

    if canonical and _normalize_text(canonical) in normalized_lookup:
        return normalized_lookup[_normalize_text(canonical)]

    # Fallback for generic metric terms like "value":
    # if there is exactly one *_value column, map to it deterministically.
    if normalized in {"value", "amount"}:
        value_columns = [col for col in available_columns if _normalize_text(col).endswith("value")]
        if len(value_columns) == 1:
            return value_columns[0]

    return None


def apply_time_scope(
    df: pd.DataFrame,
    time_scope: str = "all",
    date_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, str | None]:
    if df.empty:
        return df, "all", None

    normalized_scope = _normalize_text(time_scope or "all")
    if normalized_scope in {"", "all"}:
        return df, "all", None

    candidates = date_columns or ["close_date", "date", "created_at", "execution_date"]
    date_column = next((col for col in candidates if col in df.columns), None)

    if not date_column:
        return df, "all", "Time scope ignored because no date column is available for this dataset."

    if normalized_scope in {"quarter", "q", "qtr", "current quarter"}:
        return filter_current_quarter(df, date_column=date_column), "quarter", None

    if normalized_scope in {"year", "annual", "ytd", "current year"}:
        return filter_current_year(df, date_column=date_column), "year", None

    if normalized_scope in {"last_year", "previous_year", "prior_year", "last year", "previous year"}:
        return filter_last_year(df, date_column=date_column), "last_year", None

    return df, "all", f"Unsupported time_scope '{time_scope}'. Defaulted to all."


def apply_filters(
    df: pd.DataFrame,
    filters: list[dict[str, Any]] | None,
    field_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    if df.empty or not filters:
        return df, [], []

    warnings: list[str] = []
    applied_filters: list[dict[str, Any]] = []
    working = df.copy()

    for raw_filter in filters:
        if not isinstance(raw_filter, dict):
            warnings.append("Ignored a filter because it is not an object.")
            continue

        raw_field = str(raw_filter.get("field", "")).strip()
        op = _normalize_text(raw_filter.get("operator") or raw_filter.get("op") or "eq")
        operator_aliases = {
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
        op = operator_aliases.get(op, op)
        value = raw_filter.get("value")

        resolved_field = resolve_field_name(raw_field, list(working.columns), field_aliases)
        if not resolved_field:
            warnings.append(f"Ignored filter on unknown field '{raw_field}'.")
            continue

        series = working[resolved_field]

        if _normalize_text(resolved_field) == "sector" and op in {"eq", "=", "is", "contains", "like", "in"}:
            requested_values = value if (op == "in" and isinstance(value, list)) else [value]
            sector_frames: list[pd.DataFrame] = []
            sector_notes: list[str] = []

            for requested in requested_values:
                sector_df, sector_note = _apply_sector_filter_with_related(working, str(requested))
                if not sector_df.empty:
                    sector_frames.append(sector_df)
                if sector_note:
                    sector_notes.append(sector_note)

            if sector_frames:
                working = pd.concat(sector_frames, ignore_index=False).drop_duplicates()
            else:
                working = working.iloc[0:0]

            if sector_notes:
                warnings.extend(sorted(set(sector_notes)))

            applied_filters.append({"field": resolved_field, "operator": op, "value": value})
            continue

        if op in {"eq", "=", "is"}:
            mask = series.astype(str).str.lower() == str(value).lower()
        elif op in {"ne", "!=", "not"}:
            mask = series.astype(str).str.lower() != str(value).lower()
        elif op in {"contains", "like"}:
            mask = series.astype(str).str.contains(re.escape(str(value)), case=False, na=False)
        elif op in {"not_contains", "not like"}:
            mask = ~series.astype(str).str.contains(re.escape(str(value)), case=False, na=False)
        elif op in {"in"}:
            values = value if isinstance(value, list) else str(value).split(",")
            normalized_values = {str(v).strip().lower() for v in values if str(v).strip()}
            mask = series.astype(str).str.lower().isin(normalized_values)
        elif op in {"gt", ">", "gte", ">=", "lt", "<", "lte", "<="}:
            numeric_series = _coerce_numeric_series(working, resolved_field)
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                warnings.append(f"Ignored filter '{raw_field} {op} {value}' because value is not numeric.")
                continue
            if op in {"gt", ">"}:
                mask = numeric_series > numeric_value
            elif op in {"gte", ">="}:
                mask = numeric_series >= numeric_value
            elif op in {"lt", "<"}:
                mask = numeric_series < numeric_value
            else:
                mask = numeric_series <= numeric_value
        elif op == "is_null":
            mask = series.isna() | (series.astype(str).str.strip() == "")
        elif op == "is_not_null":
            mask = ~(series.isna() | (series.astype(str).str.strip() == ""))
        else:
            warnings.append(f"Ignored unsupported operator '{op}' on field '{raw_field}'.")
            continue

        working = working[mask]
        applied_filters.append({"field": resolved_field, "operator": op, "value": value})

    return working, warnings, applied_filters


def aggregate(
    df: pd.DataFrame,
    metrics: list[dict[str, Any]] | None,
    group_by: list[str] | None,
    field_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    metric_specs = metrics if metrics else [{"agg": "count", "field": "*", "alias": "row_count"}]

    resolved_group_by: list[str] = []
    for group_field in (group_by or []):
        resolved = resolve_field_name(str(group_field), list(df.columns), field_aliases)
        if resolved:
            resolved_group_by.append(resolved)
        else:
            warnings.append(f"Ignored unknown group_by field '{group_field}'.")

    resolved_metrics: list[dict[str, Any]] = []
    for metric in metric_specs:
        if not isinstance(metric, dict):
            warnings.append("Ignored a metric because it is not an object.")
            continue

        agg = _normalize_text(metric.get("agg") or metric.get("aggregation") or "count")
        requested_field = str(metric.get("field", "*")).strip() or "*"
        resolved_field = resolve_field_name(requested_field, list(df.columns), field_aliases) if requested_field != "*" else "*"

        if requested_field != "*" and not resolved_field:
            warnings.append(f"Ignored metric on unknown field '{requested_field}'.")
            continue

        if agg not in {"count", "sum", "avg", "mean", "min", "max"}:
            warnings.append(f"Ignored unsupported aggregation '{agg}'.")
            continue

        normalized_agg = "avg" if agg == "mean" else agg
        alias = str(metric.get("alias") or f"{resolved_field}_{normalized_agg}")
        resolved_metrics.append({"field": resolved_field, "agg": normalized_agg, "alias": alias})

    if not resolved_metrics:
        resolved_metrics = [{"field": "*", "agg": "count", "alias": "row_count"}]

    if df.empty:
        empty_columns = resolved_group_by + [m["alias"] for m in resolved_metrics]
        return pd.DataFrame(columns=empty_columns), warnings, resolved_metrics, resolved_group_by

    if resolved_group_by:
        grouped = df.groupby(resolved_group_by, dropna=False)
        output = grouped.size().reset_index(name="_group_size").drop(columns=["_group_size"])

        for metric in resolved_metrics:
            field = metric["field"]
            agg = metric["agg"]
            alias = metric["alias"]

            if agg == "count":
                if field == "*":
                    metric_df = grouped.size().reset_index(name=alias)
                else:
                    metric_df = grouped[field].count().reset_index(name=alias)
            else:
                temp_df = df.copy()
                temp_df["__metric_value"] = _coerce_numeric_series(temp_df, str(field))
                agg_map = {"sum": "sum", "avg": "mean", "min": "min", "max": "max"}
                metric_df = (
                    temp_df.groupby(resolved_group_by, dropna=False)["__metric_value"]
                    .agg(agg_map[agg])
                    .reset_index(name=alias)
                )

            output = output.merge(metric_df, on=resolved_group_by, how="left")

        return output, warnings, resolved_metrics, resolved_group_by

    row: dict[str, Any] = {}
    for metric in resolved_metrics:
        field = metric["field"]
        agg = metric["agg"]
        alias = metric["alias"]

        if agg == "count":
            row[alias] = int(len(df)) if field == "*" else int(df[str(field)].count())
        else:
            row[alias] = _aggregate_numeric(_coerce_numeric_series(df, str(field)), agg)

    return pd.DataFrame([row]), warnings, resolved_metrics, resolved_group_by


def _aggregate_numeric(series: pd.Series, agg: str) -> float:
    if series.empty:
        return 0.0
    if agg == "sum":
        return float(series.sum())
    if agg == "avg":
        return float(series.mean())
    if agg == "min":
        return float(series.min())
    if agg == "max":
        return float(series.max())
    return float(series.count())


def sort_limit(
    df: pd.DataFrame,
    sort_by: str | None,
    sort_order: str = "desc",
    limit: int = 20,
    field_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    warnings: list[str] = []
    working = df.copy()

    resolved_sort_by: str | None = None
    if sort_by:
        resolved_sort_by = resolve_field_name(sort_by, list(working.columns), field_aliases)
        if not resolved_sort_by:
            warnings.append(f"Ignored unknown sort_by field '{sort_by}'.")

    if resolved_sort_by:
        ascending = _normalize_text(sort_order) == "asc"
        working = working.sort_values(by=resolved_sort_by, ascending=ascending, kind="stable")

    safe_limit = max(1, min(int(limit or 20), 500))
    return working.head(safe_limit), warnings, resolved_sort_by


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    safe_df = df.where(pd.notna(df), None)
    records: list[dict[str, Any]] = []

    for row in safe_df.to_dict(orient="records"):
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "isoformat"):
                safe_row[key] = value.isoformat()
            elif isinstance(value, (pd.Timestamp, datetime)):
                safe_row[key] = value.isoformat()
            else:
                safe_row[key] = value
        records.append(safe_row)

    return records


def execute_bi_query(
    df: pd.DataFrame,
    *,
    filters: list[dict[str, Any]] | None = None,
    metrics: list[dict[str, Any]] | None = None,
    group_by: list[str] | None = None,
    time_scope: str = "all",
    sort_by: str | None = None,
    sort_order: str = "desc",
    limit: int = 20,
    field_aliases: dict[str, str] | None = None,
    date_columns: list[str] | None = None,
) -> dict[str, Any]:
    base_count = int(len(df))

    timed_df, applied_scope, time_warning = apply_time_scope(
        df,
        time_scope=time_scope,
        date_columns=date_columns,
    )

    filtered_df, filter_warnings, applied_filters = apply_filters(
        timed_df,
        filters=filters,
        field_aliases=field_aliases,
    )

    aggregated_df, metric_warnings, resolved_metrics, resolved_group_by = aggregate(
        filtered_df,
        metrics=metrics,
        group_by=group_by,
        field_aliases=field_aliases,
    )

    sorted_df, sort_warnings, resolved_sort_by = sort_limit(
        aggregated_df,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        field_aliases=field_aliases,
    )

    warnings = [*filter_warnings, *metric_warnings, *sort_warnings]
    if time_warning:
        warnings.append(time_warning)

    return {
        "row_count_before_time_scope": base_count,
        "row_count_after_time_scope": int(len(timed_df)),
        "row_count_after_filters": int(len(filtered_df)),
        "time_scope_applied": applied_scope,
        "applied_filters": applied_filters,
        "group_by_applied": resolved_group_by,
        "metrics_applied": resolved_metrics,
        "sort_by_applied": resolved_sort_by,
        "result_count": int(len(sorted_df)),
        "result": dataframe_to_records(sorted_df),
        "warnings": warnings,
        "available_columns": list(df.columns),
    }


# ==========================================================
# FORMAT CURRENCY
# ==========================================================


def format_currency(value: float) -> str:
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:.2f}"
