#!/usr/bin/env python3
"""Add a categorical outcome_label column to GlobalProtestTracker.csv.

This script applies deterministic regex rules (priority order) to map the free-text
`Outcomes` column into one of the target categories and writes the processed CSV to
data/processed/GlobalProtestTracker_with_outcomes.csv.

Categories (priority order):
 - regime shift
 - Policy changed to meet demands (fully changed/reversed)
 - Partial policy change
 - partial political change
 - No significant change

Run from repository root:
    python scripts/add_outcome_category.py
"""
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_IN = ROOT / "data" / "raw" / "GlobalProtestTracker.csv"
DATA_OUT = ROOT / "data" / "processed" / "GlobalProtestTracker_with_outcomes.csv"

LABELS = [
    "regime shift",
    "Policy changed to meet demands (fully changed/reversed)",
    "Partial policy change",
    "partial political change",
    "No significant change",
]

# Priority-ordered rules (label, list of regex patterns)
RULES = [
    ("regime shift", [
        r"\b(resign(?:ed|s)? as|resignation of the (?:prime minister|president|leader)|ousted|overthrew|military coup|took power|new government|called (?:early )?elections|announced elections|interim government|removed from office)\b",
        r"\b(new (?:government|prime minister|president)|appointed a new prime minister|installed a new government|was forced out of office|was forced to step down|was removed)\b",
    ]),
    ("Policy changed to meet demands (fully changed/reversed)", [
        r"\b(agreed to|accepted demands|adopted|rescinded|reversed|repealed|suspended (?:the )?(?:law|policy)|cancelled|implemented the reform|signed .* into law|withdrew the bill|adopted .* reform|withdrawn the proposal)\b",
    ]),
    ("Partial policy change", [
        r"\b(partial|limited|some concessions|scaled back|amend(?:ed|s)? (?:the )?(?:law|policy)|modified|tweaked|revised policy)\b",
        r"\b(suspend(?:ed)? (?:temporarily|for now)|temporary suspension|scaled back)\b",
    ]),
    ("partial political change", [
        r"\b(minister(?:s)? resigned|cabinet (?:minister)?s? resigned|resignation of a minister|resignations of ministers|one minister resigned|official resigned)\b",
        r"\b(appointed a new minister|reshuffle|cabinet reshuffle|new cabinet member|dismissed the minister)\b",
    ]),
    ("No significant change", [
        r"\b(no policy change|no leadership change|no change in response|no policy or leadership change|no effect|no response|no policy/leadership change)\b",
    ]),
]

def rule_based_label(text: str):
    if not isinstance(text, str) or not text.strip():
        return "No significant change"
    t = text.lower()
    for label, patterns in RULES:
        for p in patterns:
            if re.search(p, t):
                return label
    # fallback heuristics
    # common short phrases mapping
    if any(tok in t for tok in ["resign", "resigned", "resignation"]):
        # if resignation of head (prime minister/president) appears earlier regex should capture regime shift
        return "partial political change"
    if any(tok in t for tok in ["reversed", "repealed", "rescinded", "withdrew", "cancelled", "suspended"]):
        return "Policy changed to meet demands (fully changed/reversed)"
    return "No significant change"

def main():
    if not DATA_IN.exists():
        print(f"Input file not found: {DATA_IN}")
        sys.exit(1)
    df = pd.read_csv(DATA_IN)
    if 'Outcomes' not in df.columns:
        print("No 'Outcomes' column in CSV")
        sys.exit(1)

    df['outcome_label'] = df['Outcomes'].fillna("").map(rule_based_label)

    out_dir = DATA_OUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_OUT, index=False)
    print(f"Wrote {DATA_OUT}")
    print("Label distribution:\n")
    print(df['outcome_label'].value_counts(dropna=False))

if __name__ == '__main__':
    main()
