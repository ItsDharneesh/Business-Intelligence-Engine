import json

import pandas as pd


def normalize_items(raw_json):
    boards = raw_json.get("data", {}).get("boards", [])

    if not boards:
        return pd.DataFrame()

    items = boards[0].get("items_page", {}).get("items", [])
    if not items:
        return pd.DataFrame()

    rows = []

    for item in items:
        row = {
            "item_id": item.get("id"),
            "item_name": item.get("name"),
        }

        for col in item.get("column_values", []):
            col_id = col.get("id")
            text_value = col.get("text")
            raw_value = col.get("value")

            value = None

            if text_value:
                value = text_value
            elif raw_value:
                try:
                    parsed = json.loads(raw_value)
                    if isinstance(parsed, dict):
                        value = (
                            parsed.get("text")
                            or parsed.get("label")
                            or parsed.get("date")
                            or parsed.get("number")
                            or str(parsed)
                        )
                    else:
                        value = str(parsed)
                except Exception:
                    value = str(raw_value)

            row[col_id] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df.replace(["", "None", "null", "NaN"], pd.NA, inplace=True)

    column_map = {
        "numeric_mm118js8": "deal_value",
        "color_mm11caxn": "status",
        "date_mm11vs0e": "close_date",
        "color_mm113yep": "sector",
        "color_mm1187gn": "owner",
        "color_mm11vyga": "sector",
        "color_mm11jc1s": "execution_status",
        "numeric_mm11p75q": "wo_value",
    }

    df = df.rename(columns=column_map)
    return df
