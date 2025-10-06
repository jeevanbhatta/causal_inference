"""Visualization helpers for the analysis notebook."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def _ensure_outdir(out_path: Path):
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)


def plot_duration_hist(df: pd.DataFrame, out_path: Path = None):
    plt.figure(figsize=(8, 4))
    sns.histplot(df["duration_days"].dropna(), bins=50)
    plt.title("Distribution of protest duration (days)")
    if out_path:
        _ensure_outdir(out_path)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_by_country_counts(df: pd.DataFrame, country_col: str = "country", out_path: Path = None, top_n=20):
    if country_col not in df.columns:
        raise ValueError(f"Country column {country_col} not found in dataframe")
    counts = df[country_col].value_counts().nlargest(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index)
    plt.xlabel("Number of events")
    plt.title("Top countries by number of protest events")
    if out_path:
        _ensure_outdir(out_path)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_scatter_with_reg(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path = None, hue: str = None):
    """Scatter plot with regression line (for numeric x and y)."""
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Columns not found in dataframe")
    plt.figure(figsize=(8, 6))
    if hue and hue in df.columns:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col)
    # draw regression if both numeric
    try:
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, truncate=True, color="red")
    except Exception:
        pass
    plt.title(f"{y_col} vs {x_col}")
    if out_path:
        _ensure_outdir(out_path)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list = None, out_path: Path = None, method: str = "spearman"):
    """Plot a correlation heatmap for numeric columns.

    method: spearman|pearson
    """
    if numeric_cols is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df.loc[:, [c for c in numeric_cols if c in df.columns]]
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found for correlation")
    corr = numeric_df.corr(method=method)
    plt.figure(figsize=(max(6, 0.5 * corr.shape[0]), max(4, 0.5 * corr.shape[1])))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
    plt.title(f"Correlation matrix ({method})")
    if out_path:
        _ensure_outdir(out_path)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def top_outcomes_bar(df: pd.DataFrame, outcomes_col: str = 'Outcomes', top_n: int = 20, out_path: Path = None):
    if outcomes_col not in df.columns:
        raise ValueError(f"Outcomes column {outcomes_col} not found")
    counts = df[outcomes_col].fillna('Unknown').value_counts().nlargest(top_n)
    plt.figure(figsize=(10, max(4, 0.3 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title('Top outcomes')
    if out_path:
        _ensure_outdir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def boxplot_duration_by_outcome(df: pd.DataFrame, outcomes_col: str = 'Outcomes', top_n: int = 10, out_path: Path = None):
    if outcomes_col not in df.columns:
        raise ValueError(f"Outcomes column {outcomes_col} not found")
    top = df[outcomes_col].fillna('Unknown').value_counts().nlargest(top_n).index.tolist()
    subset = df[df[outcomes_col].isin(top)].copy()
    plt.figure(figsize=(10, max(4, 0.6 * len(top))))
    sns.boxplot(x='duration_days', y=outcomes_col, data=subset)
    plt.title('Duration (days) by Outcome (top categories)')
    if out_path:
        _ensure_outdir(out_path)
        plt.savefig(out_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print('visualize.py - helper functions for plotting (not a runnable script)')
