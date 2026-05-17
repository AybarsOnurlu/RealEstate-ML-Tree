"""
seed_supabase.py — King County House Sales Dataset → Supabase `properties` table
==============================================================================
Downloads the Kaggle dataset and bulk-inserts all ~21,613 records.

Target Supabase table schema:
  id          UUID (auto-generated)
  price       FLOAT8
  bedrooms    FLOAT8
  bathrooms   FLOAT8
  sqft_living FLOAT8
  sqft_lot    FLOAT8
  floors      FLOAT8
  yr_built    FLOAT8
  lat         FLOAT8
  long        FLOAT8

Usage:
  python seed_supabase.py
  python seed_supabase.py --chunk-size 500   (smaller batches if needed)
  python seed_supabase.py --dry-run          (validate without uploading)
  python seed_supabase.py --truncate         (clear table before seeding)
"""

import sys
# Force UTF-8 output on Windows (avoids cp1254 charmap errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import math
import os

import kagglehub
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")


# ── CLI arguments ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed Supabase `properties` table with King County House Sales data."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Number of rows per batch insert (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download and transform data but do NOT write to Supabase",
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="Delete all existing rows before inserting (fresh seed)",
    )
    return parser.parse_args()


# ── Dataset download ──────────────────────────────────────────────────────────

def download_dataset() -> pd.DataFrame:
    """Download the Kaggle dataset and return it as a DataFrame."""
    print("[INFO] Downloading King County House Sales dataset from Kaggle ...")
    dataset_path = kagglehub.dataset_download("harlfoxem/housesalesprediction")
    csv_path = os.path.join(dataset_path, "kc_house_data.csv")

    if not os.path.exists(csv_path):
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".csv"):
                    csv_path = os.path.join(root, f)
                    break

    print(f"[OK]   Dataset found at: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[OK]   Loaded {len(df):,} rows, {len(df.columns)} columns.")
    print(f"       Columns: {list(df.columns)}\n")
    return df


# ── Column mapping & transformation ──────────────────────────────────────────

def transform(df: pd.DataFrame) -> list[dict]:
    """
    Clean the dataset and map directly to the Supabase schema.
    """
    print("[INFO] Transforming data ...")

    # Drop any rows with essential missing values (if any)
    original_count = len(df)
    df = df.dropna(subset=["price", "lat", "long"]).copy()
    dropped = original_count - len(df)
    if dropped:
        print(f"[WARN]  Dropped {dropped} rows with null target or location values.")

    records = []
    for _, row in df.iterrows():
        records.append({
            "price":       float(row["price"]),
            "bedrooms":    float(row["bedrooms"]),
            "bathrooms":   float(row["bathrooms"]),
            "sqft_living": float(row["sqft_living"]),
            "sqft_lot":    float(row["sqft_lot"]),
            "floors":      float(row["floors"]),
            "yr_built":    float(row["yr_built"]),
            "lat":         round(float(row["lat"]), 6),
            "long":        round(float(row["long"]), 6),
        })

    print(f"[OK]   {len(records):,} records ready for upload.\n")
    return records


# ── Batch upload ──────────────────────────────────────────────────────────────

def upload(records: list[dict], supabase: Client, chunk_size: int) -> None:
    """Insert records into Supabase in chunks, printing progress."""
    total    = len(records)
    uploaded = 0
    failed_chunks = 0

    print(f"[INFO] Uploading {total:,} records in chunks of {chunk_size} ...\n")

    for start in range(0, total, chunk_size):
        chunk = records[start: start + chunk_size]
        end   = min(start + chunk_size, total)

        try:
            supabase.table("properties").insert(chunk).execute()
            uploaded += len(chunk)
            bar_filled = math.floor((uploaded / total) * 30)
            bar = "█" * bar_filled + "░" * (30 - bar_filled)
            pct = uploaded / total * 100
            print(
                f"  [{bar}] {pct:5.1f}%  —  Uploaded {uploaded:,}/{total:,} records",
                end="\r", flush=True,
            )
        except Exception as exc:
            failed_chunks += 1
            print(f"\n  [ERR]  Chunk {start}-{end} failed: {exc}", file=sys.stderr)
            if failed_chunks >= 5:
                print("\n  [ABORT] Too many failures. Aborting.", file=sys.stderr)
                sys.exit(1)

    print(f"\n\n[DONE] {uploaded:,} records inserted into `properties`.")
    if failed_chunks:
        print(f"[WARN]  {failed_chunks} chunk(s) failed — check logs above.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.dry_run:
        missing = [
            k for k, v in {"SUPABASE_URL": SUPABASE_URL, "SUPABASE_KEY": SUPABASE_KEY}.items()
            if not v
        ]
        if missing:
            print(
                f"[ERR]  Missing environment variable(s): {missing}\n"
                "       Copy .env.example -> .env and fill in your Supabase credentials.",
                file=sys.stderr,
            )
            sys.exit(1)

    df      = download_dataset()
    records = transform(df)

    if args.dry_run:
        print("[DRY RUN] First 3 transformed records:")
        for r in records[:3]:
            print(f"    {r}")
        print("\n[DRY RUN] Skipping upload (--dry-run mode).")
        return

    print(f"[INFO] Connecting to Supabase: {SUPABASE_URL[:50]}...\n")
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    if args.truncate:
        print("[INFO] --truncate: deleting all existing rows from `properties` ...")
        supabase_client.table("properties").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("[OK]   Table cleared.\n")

    upload(records, supabase_client, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
