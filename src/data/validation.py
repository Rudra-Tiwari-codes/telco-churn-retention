"""
Great Expectations based validation for the Telco churn dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from great_expectations.core.batch import Batch
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.data_context import get_context, set_context
from great_expectations.execution_engine.pandas_execution_engine import (
    PandasExecutionEngine,
)
from great_expectations.validator.validator import Validator


def build_validator(df: pd.DataFrame) -> Validator:
    context = get_context()
    set_context(context)
    suite = ExpectationSuite(name="telco_phase1")
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[Batch(data=df)],
        expectation_suite=suite,
        data_context=context,
    )
    validator.expect_column_values_to_not_be_null("customerID")
    validator.expect_column_values_to_be_unique("customerID")
    validator.expect_table_column_count_to_equal(21)
    validator.expect_column_values_to_be_in_set(
        "Churn",
        ["Yes", "No"],
    )
    validator.expect_column_values_to_match_regex_list(
        "customerID",
        regex_list=[r"^[0-9A-Za-z\-]+$"],
    )
    validator.expect_column_values_to_not_be_null("TotalCharges", mostly=0.995)
    validator.expect_column_values_to_be_between(
        "MonthlyCharges", min_value=0, max_value=200
    )
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=100)
    return validator


def run_validation(df: pd.DataFrame, output_path: Path) -> dict:
    validator = build_validator(df)
    results = validator.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results.to_json_dict(), indent=2))
    return results

