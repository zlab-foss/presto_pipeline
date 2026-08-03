"""
Concatenate the per-worker manifest CSVs written by get_data.py into a
single training manifest, then split it into train/val by parcel.

Each worker CSV has one row per (parcel, candidate crop, source) tile:
path, source, actual_crop, candidate_crop, target, parcel_id, poly_idx,
tile_tag, province, plant, harvest, season_days, start_month.

Steps:
  1. Concatenate all worker CSVs under --manifest-dir.
  2. Drop rows whose tif is missing or unreadable (_is_valid_tif).
  3. Dedupe on (parcel_id, tile_tag, candidate_crop, source) -- resumed
     runs can rewrite a worker's CSV, so later rows win.
  4. Pivot source: each output row is one training example with both
     s2_path and s1_path; rows missing either source are dropped.
  5. Split into train/val by parcel_id (an 80/20 split by field would leak
     a parcel's positive and negative candidate rows across the split).
  6. Write manifest.csv (full), manifest_train.csv, manifest_val.csv, and
     print a summary (row counts, positives, per-candidate counts).

Run: python scripts/build_manifest.py --manifest-dir ./data/manifest --data-root ./data
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from get_data import _is_valid_tif  # noqa: E402

MERGE_KEY = ["parcel_id", "tile_tag", "candidate_crop"]
SHARED_COLS = [
    "actual_crop",
    "target",
    "poly_idx",
    "province",
    "plant",
    "harvest",
    "season_days",
    "start_month",
]


def load_worker_csvs(manifest_dir: Path) -> pd.DataFrame:
    csv_files = sorted(manifest_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No manifest CSVs found under {manifest_dir}")
    print(f"Found {len(csv_files)} worker manifest CSV(s) in {manifest_dir}")
    frames = [pd.read_csv(f, dtype={"tile_tag": str, "province": str}) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Concatenated rows: {len(df)}")
    return df


def drop_invalid_tifs(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    valid = df["path"].apply(lambda p: _is_valid_tif(data_root / p))
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} rows with missing/unreadable tifs")
    return df[valid].reset_index(drop=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = MERGE_KEY + ["source"]
    n_before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} duplicate rows on {key_cols}")
    return df


def pivot_sources(df: pd.DataFrame) -> pd.DataFrame:
    s2 = df[df["source"] == "s2"].rename(columns={"path": "s2_path"})
    s2 = s2[MERGE_KEY + ["s2_path"] + SHARED_COLS]

    s1 = df[df["source"] == "s1"].rename(columns={"path": "s1_path"})
    s1 = s1[MERGE_KEY + ["s1_path"]]

    wide = s2.merge(s1, on=MERGE_KEY, how="inner")
    n_s2, n_s1, n_wide = len(s2), len(s1), len(wide)
    print(f"  s2 rows: {n_s2}, s1 rows: {n_s1}, paired (both sources) rows: {n_wide}")
    return wide


def split_train_val(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    parcel_ids = sorted(df["parcel_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(parcel_ids)

    n_val = max(1, int(len(parcel_ids) * val_frac)) if parcel_ids else 0
    val_ids = set(parcel_ids[:n_val])
    train_ids = set(parcel_ids[n_val:])

    train_df = df[df["parcel_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["parcel_id"].isin(val_ids)].reset_index(drop=True)

    overlap = set(train_df["parcel_id"]) & set(val_df["parcel_id"])
    assert not overlap, f"parcel_id leaked across train/val split: {overlap}"

    return train_df, val_df


def print_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    print(f"Rows       : {len(df)}")
    if len(df) == 0:
        return
    print(f"Parcels    : {df['parcel_id'].nunique()}")
    print(f"Positives  : {int(df['target'].sum())} ({df['target'].mean():.1%})")
    print("Per-candidate counts:")
    counts = df.groupby("candidate_crop")["target"].agg(["count", "sum"])
    for candidate, row in counts.iterrows():
        print(f"  {candidate:<8s} total={int(row['count']):>7d}  positives={int(row['sum']):>6d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", default="./data/manifest", type=str)
    p.add_argument("--data-root", default="./data", type=str, help="Root that manifest 'path' columns are relative to")
    p.add_argument("--out", default="./data/manifest.csv", type=str, help="Full pivoted/deduped manifest output")
    p.add_argument("--val-frac", default=0.1, type=float)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    manifest_dir = Path(args.manifest_dir)
    data_root = Path(args.data_root)
    out_path = Path(args.out)

    print("=== Loading worker manifests ===")
    df = load_worker_csvs(manifest_dir)

    print("\n=== Validating tifs ===")
    df = drop_invalid_tifs(df, data_root)

    print("\n=== Deduping ===")
    df = dedupe(df)

    print("\n=== Pivoting source -> s2_path/s1_path ===")
    wide = pivot_sources(df)

    print("\n=== Splitting train/val by parcel_id ===")
    train_df, val_df = split_train_val(wide, args.val_frac, args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    train_path = out_path.with_name(f"{out_path.stem}_train.csv")
    val_path = out_path.with_name(f"{out_path.stem}_val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\nWrote {out_path} ({len(wide)} rows)")
    print(f"Wrote {train_path} ({len(train_df)} rows)")
    print(f"Wrote {val_path} ({len(val_df)} rows)")

    print_summary("Train", train_df)
    print_summary("Val", val_df)


if __name__ == "__main__":
    main()
