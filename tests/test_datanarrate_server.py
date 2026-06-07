import numpy as np
import pandas as pd

import datanarrate_server as server


def test_parse_csv_env_uses_default_for_missing_value(monkeypatch):
    monkeypatch.delenv("DATANARRATE_TEST_LIST", raising=False)

    assert server.parse_csv_env("DATANARRATE_TEST_LIST", ["http://localhost"]) == ["http://localhost"]


def test_parse_csv_env_trims_and_drops_empty_values(monkeypatch):
    monkeypatch.setenv("DATANARRATE_TEST_LIST", " http://a.example, ,http://b.example ")

    assert server.parse_csv_env("DATANARRATE_TEST_LIST", []) == ["http://a.example", "http://b.example"]


def test_safe_cleaning_normalizes_columns_missing_tokens_and_duplicates():
    df = pd.DataFrame(
        {
            "Customer Name": [" Ada ", "Ada", "NULL"],
            "Order Total": ["10", "10", "20"],
        }
    )

    cleaned, log = server.apply_safe_cleaning(df)

    assert list(cleaned.columns) == ["customer_name", "order_total"]
    assert len(cleaned) == 2
    assert cleaned["order_total"].tolist() == [10, 20]
    assert cleaned["customer_name"].isna().sum() == 1
    assert {entry["action"] for entry in log} >= {
        "normalize_column_names",
        "trim_whitespace",
        "normalize_missing_tokens",
        "remove_duplicates",
        "safe_cleaning_complete",
    }


def test_profile_dataframe_reports_core_shape_and_column_types():
    df = pd.DataFrame(
        {
            "amount": [10.0, 12.0, np.nan],
            "segment": ["A", "B", "B"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )

    profile = server.profile_dataframe(df)

    assert profile["n_rows"] == 3
    assert profile["n_cols"] == 3
    assert profile["duplicates_count"] == 0
    assert "amount" in profile["numeric_cols"]
    assert "segment" in profile["categorical_cols"]
    assert "date" in profile["datetime_cols"]
    assert "Dataset has 3 rows and 3 columns." in profile["summary_text"]
