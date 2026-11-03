"""Load BeIR/hotpotqa corpus into the RAG system via pandas + parquet.

Usage:
    python scripts/load_dataset.py                  # default: 500 docs
    python scripts/load_dataset.py --limit 1000
"""

import argparse

import httpx
import pandas as pd


API_URL = "http://localhost:8000"
CORPUS_URL = "hf://datasets/BeIR/hotpotqa/corpus/corpus-00000-of-00001.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading hotpotqa corpus (limit={args.limit})...")
    df = pd.read_parquet(CORPUS_URL)

    # Filter out very short texts and take the limit
    df = df[df["text"].str.len() > 50].head(args.limit)
    print(f"Got {len(df)} documents")

    total = 0
    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i : i + args.batch_size]
        documents = [
            {"text": row["text"], "source": f"hotpotqa:{row['title']}"}
            for _, row in batch.iterrows()
        ]
        resp = httpx.post(
            f"{API_URL}/ingest/bulk",
            json={"documents": documents},
            timeout=120,
        )
        resp.raise_for_status()
        count = resp.json()["chunks_ingested"]
        total += count
        print(f"  Batch {i // args.batch_size + 1}: {count} chunks ingested")

    print(f"Done! Total chunks: {total}")


if __name__ == "__main__":
    main()
