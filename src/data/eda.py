"""
Lightweight exploratory data analysis utilities for Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def churn_distribution(df: pd.DataFrame) -> str:
    dist = df["Churn"].value_counts(normalize=False).rename_axis("label").reset_index(name="count")
    dist["pct"] = (dist["count"] / len(df)).round(3)
    rows = "\n".join(
        f"| {row['label']} | {int(row['count'])} | {row['pct']:.3f} |" for _, row in dist.iterrows()
    )
    header = "| Label | Count | Share |\n| --- | --- | --- |"
    return "\n".join([header, rows])


def missingness_table(df: pd.DataFrame) -> str:
    miss = df.isna().mean().sort_values(ascending=False)
    rows = "\n".join(f"| {idx} | {pct:.3f} |" for idx, pct in miss.items() if pct > 0)
    if not rows:
        rows = "| (none) | 0.000 |"
    header = "| Column | Missing Share |\n| --- | --- |"
    return "\n".join([header, rows])


def numeric_summary(df: pd.DataFrame) -> str:
    desc = df.select_dtypes(include="number").describe().T.round(3)
    rows = "\n".join(
        f"| {idx} | {row['mean']} | {row['std']} | {row['min']} | {row['50%']} | {row['max']} |"
        for idx, row in desc.iterrows()
    )
    header = "| Column | Mean | Std | Min | Median | Max |\n| --- | --- | --- | --- | --- | --- |"
    return "\n".join([header, rows])


def categorical_cardinality(df: pd.DataFrame) -> str:
    cat_cols = df.select_dtypes(include="object").columns
    lines = [f"| {col} | {df[col].nunique()} |" for col in cat_cols if df[col].nunique() <= 50]
    if not lines:
        lines = ["| (none) | 0 |"]
    header = "| Column | Unique Values |\n| --- | --- |"
    return "\n".join([header, "\n".join(lines)])


def render_markdown_summary(df: pd.DataFrame) -> str:
    lines = [
        "# Phase 1 EDA Snapshot",
        "",
        "## Dataset Overview",
        f"- Rows: {len(df):,}",
        f"- Columns: {df.shape[1]}",
        "",
        "## Churn Distribution",
        churn_distribution(df),
        "",
        "## Numeric Summary",
        numeric_summary(df),
        "",
        "## Missingness",
        missingness_table(df),
        "",
        "## Categorical Cardinality (<=50 uniques)",
        categorical_cardinality(df),
    ]
    return "\n".join(lines)


def write_markdown_summary(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_summary(df))
    return output_path
