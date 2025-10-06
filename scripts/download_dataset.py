"""Download dataset from Kaggle (uses kaggle API)."""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()
    # Use KaggleApi class to separate import from authentication errors
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError:
        print("kaggle package not found. Install it with: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        # Likely missing ~/.kaggle/kaggle.json or env vars
        print("Kaggle authentication failed:", e)
        print("Make sure you have your Kaggle credentials set up. Options:")
        print("  1) Place your kaggle.json in ~/.kaggle/kaggle.json (recommended).")
        print("  2) Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY.")
        print("See https://github.com/Kaggle/kaggle-api#api-credentials for details.")
        sys.exit(2)

    dataset = "kkhandekar/global-protest-tracker"
    print(f"Downloading dataset {dataset} to {RAW_DIR}")
    try:
        api.dataset_download_files(dataset, path=str(RAW_DIR), unzip=True, quiet=False)
        print("Download complete.")
    except Exception as e:
        print("Failed to download dataset:", e)
        sys.exit(3)


if __name__ == "__main__":
    main()
