"""Print the real schema and 5 example rows of the TikTok dataset using `datasets` streaming.

Streaming never downloads the full dataset: it reads just enough of one Parquet file to yield
the first few rows. We point it at a single monthly file so it doesn't have to resolve all 148.

Usage:  python -m src.inspect_schema [--month 2024-03]
"""

import argparse

from datasets import load_dataset

REPO_ID = "datasocial/tiktok-5.6B-videos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2024-03", help="monthly file to peek at (YYYY-MM)")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    ds = load_dataset(
        REPO_ID,
        data_files=f"data/{args.month}.parquet",
        split="train",
        streaming=True,
    )

    print("=== Features (schema) ===")
    for name, feature in ds.features.items():
        print(f"{name:28s} {feature}")

    print(f"\n=== First {args.n} rows of data/{args.month}.parquet ===")
    for i, row in enumerate(ds.take(args.n)):
        print(f"\n--- row {i} ---")
        for k, v in row.items():
            text = repr(v)
            print(f"  {k:26s} {text[:120]}")


if __name__ == "__main__":
    main()
