from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.schema_catalog import constrain_run_bi_query_arguments, detect_health_intent
from agent.tool_registry import compute_health_from_status_rows
from services.bi_engine import execute_bi_query


def test_constrained_planner_args() -> None:
    raw_args = {
        "dataset": "Deals",
        "time_scope": "QTR",
        "filters": [
            {"field": "Industry", "operator": "equals", "value": "energy"},
            {"field": "owner", "operator": "greater_than", "value": "x"},
        ],
        "metrics": [
            {"field": "Projected_Revenue", "agg": "SUM", "alias": "projected_revenue"},
            {"field": "*", "agg": "COUNT", "alias": "deals"},
        ],
        "group_by": ["Stage"],
        "sort_by": "Revenue",
        "sort_order": "DESC",
        "limit": "5000",
    }
    constrained, _ = constrain_run_bi_query_arguments(raw_args, "How is energy sector doing this quarter?")

    assert constrained["dataset"] == "deals"
    assert constrained["time_scope"] in {"all", "quarter", "year", "last_year"}
    assert constrained["filters"][0]["field"] == "sector"
    assert constrained["filters"][0]["operator"] == "eq"
    assert constrained["group_by"][0] == "status"
    assert constrained["sort_by"] == "deal_value"
    assert constrained["limit"] == 200


def test_health_scoring() -> None:
    status_rows = [
        {"status": "Won", "status_value": 8_000_000, "deal_count": 30},
        {"status": "Open", "status_value": 5_000_000, "deal_count": 25},
        {"status": "Dead", "status_value": 2_000_000, "deal_count": 10},
    ]
    result = compute_health_from_status_rows(status_rows, time_scope_applied="quarter")

    assert result["health_score"] >= 60
    assert result["health_flag"] in {"Healthy", "Watch"}
    assert result["totals"]["total_deal_count"] == 65


def test_sector_fallback() -> None:
    df = pd.DataFrame(
        [
            {"sector": "Renewables", "status": "Won", "deal_value": "$1000", "close_date": "2026-01-10"},
            {"sector": "Powerline", "status": "Open", "deal_value": "$2000", "close_date": "2026-01-11"},
            {"sector": "Mining", "status": "Won", "deal_value": "$3000", "close_date": "2026-01-12"},
        ]
    )

    result = execute_bi_query(
        df,
        filters=[{"field": "sector", "operator": "equals", "value": "energy"}],
        metrics=[{"field": "deal_value", "agg": "sum", "alias": "value_sum"}],
        group_by=["status"],
        field_aliases={"stage": "status", "value": "deal_value"},
        date_columns=["close_date"],
    )

    warnings = result.get("warnings") or []
    assert any("No exact sector 'energy' found." in warning for warning in warnings)
    assert result["result_count"] > 0


def test_health_intent_detection() -> None:
    assert detect_health_intent("Are we healthy this quarter?")
    assert detect_health_intent("Are we going in the wrong direction this year?")
    assert not detect_health_intent("Top 5 owners by revenue")


if __name__ == "__main__":
    test_constrained_planner_args()
    test_health_scoring()
    test_sector_fallback()
    test_health_intent_detection()
    print("Regression smoke tests passed.")
