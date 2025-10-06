# Global Protest Tracker - causal_inference

This repository stores analysis for the Global Protest Tracker dataset (kkhandekar/global-protest-tracker) from Kaggle / Carnegie Endowment.

Data source: https://carnegieendowment.org/ and dataset on Kaggle: kkhandekar/global-protest-tracker (CC0)

Quick start

1. Install dependencies:

   pip install -r requirements.txt

2. Download the dataset (you must configure Kaggle credentials - set KAGGLE_USERNAME and KAGGLE_KEY or use ~/.kaggle/kaggle.json):

   python scripts/download_dataset.py

3. Run the notebook at `notebooks/analysis.ipynb` to explore the data.

Files added:
- `scripts/download_dataset.py`: download helper
- `src/etl.py`: ETL helper to load CSVs and add derived columns
- `src/visualize.py`: simple plotting helpers
- `notebooks/analysis.ipynb`: starter notebook for exploration
- `requirements.txt`: minimal dependencies

License: CC0 (per dataset license). This repo is for coursework.
