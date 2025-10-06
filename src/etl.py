"""ETL utilities for the Global Protest Tracker dataset."""
from pathlib import Path
import pandas as pd


def find_csvs(data_dir: Path):
    return list(data_dir.glob("**/*.csv"))


def load_primary_csv(data_dir: Path) -> pd.DataFrame:
    files = find_csvs(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Prefer files with 'protest' or 'events' in name
    for f in files:
        name = f.name.lower()
        if "protest" in name or "events" in name or "global_protest" in name:
            return pd.read_csv(f)

    # fallback to first csv
    return pd.read_csv(files[0])


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add helpful columns: duration_days if start/end present, placeholder percent_population."""
    df = df.copy()

    # Common column names: start_date, end_date, start, end
    date_cols = {"start": ["start_date", "start", "begin_date"],
                 "end": ["end_date", "end", "finish_date"]}

    for key, candidates in date_cols.items():
        for c in candidates:
            if c in df.columns:
                df[c + "_parsed"] = pd.to_datetime(df[c], errors="coerce")
                break

    if "start_date_parsed" in df.columns and "end_date_parsed" in df.columns:
        df["duration_days"] = (df["end_date_parsed"] - df["start_date_parsed"]).dt.days
    else:
        # Try a single date column, set duration 1
        df["duration_days"] = 1

    # Placeholder percent_population: if population and affected present
    if "population" in df.columns and "affected" in df.columns:
        # avoid division by zero
        df["percent_population_affected"] = df.apply(
            lambda r: (r.get("affected") / r.get("population") * 100)
            if pd.notna(r.get("affected")) and pd.notna(r.get("population")) and r.get("population") > 0
            else None,
            axis=1,
        )
    else:
        df["percent_population_affected"] = None

    return df


def save_processed(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
