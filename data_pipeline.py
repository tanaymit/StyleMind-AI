"""
StyleMind AI - Data Pipeline
Cleans the Fashion Product Images Dataset, constructs embedding strings,
generates OpenAI embeddings, and builds the FAISS index.

Usage:
    python data_pipeline.py --csv path/to/styles.csv
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

from config import (
    CATALOG_COLUMNS,
    DATA_DIR,
    DEFAULT_PRICE_RANGE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDINGS_DIR,
    EMBEDDINGS_NPY_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    OPENAI_API_KEY,
    PRICE_RANGES,
)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and perform cleaning."""
    print(f"[1/5] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    original_len = len(df)

    # Drop rows missing critical fields
    df = df.dropna(subset=["productDisplayName", "articleType", "gender", "masterCategory"])

    # Fill optional nulls
    df["baseColour"] = df["baseColour"].fillna("Unknown")
    df["season"] = df["season"].fillna("All Season")
    df["usage"] = df["usage"].fillna("Casual")
    df["year"] = df["year"].fillna(df["year"].median())

    # Filter to clothing-relevant categories (drop Personal Care, Free Items, etc.)
    relevant_categories = ["Apparel", "Accessories", "Footwear"]
    df = df[df["masterCategory"].isin(relevant_categories)].copy()

    # Assign simulated prices based on articleType
    def assign_price(article_type: str) -> float:
        low, high = PRICE_RANGES.get(article_type, DEFAULT_PRICE_RANGE)
        return round(random.uniform(low, high), 2)

    random.seed(42)
    df["price"] = df["articleType"].apply(assign_price)

    # Reset index
    df = df.reset_index(drop=True)

    print(f"    Cleaned: {original_len} → {len(df)} items "
          f"(kept {len(df)/original_len*100:.1f}%)")

    return df


def build_embedding_strings(df: pd.DataFrame) -> list[str]:
    """Construct the embedding string for each product."""
    print("[2/5] Building embedding strings...")
    strings = []
    for _, row in df.iterrows():
        s = (
            f"{row['productDisplayName']} | {row['articleType']} | "
            f"{row['usage']} | {row['baseColour']} | {row['season']} | "
            f"{row['gender']}"
        )
        strings.append(s)
    print(f"    Built {len(strings)} embedding strings")
    return strings


def generate_embeddings(texts: list[str], batch_size: int = 500) -> np.ndarray:
    """Generate embeddings using OpenAI's API in batches."""
    print(f"[3/5] Generating embeddings with {EMBEDDING_MODEL}...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"    Batch {batch_num}/{total_batches} "
              f"({len(batch)} items)...", end=" ")

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)
            print("done")
        except Exception as e:
            print(f"\n    ERROR on batch {batch_num}: {e}")
            print("    Retrying in 10 seconds...")
            time.sleep(10)
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                batch_embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(batch_embeddings)
                print(f"    Batch {batch_num} retry succeeded")
            except Exception as e2:
                print(f"    FATAL: Batch {batch_num} failed again: {e2}")
                sys.exit(1)

        # Small delay to respect rate limits
        if batch_num < total_batches:
            time.sleep(0.5)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"    Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS index using inner product (cosine similarity on normalized vectors)."""
    print("[4/5] Building FAISS index...")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(normalized)

    print(f"    Index contains {index.ntotal} vectors "
          f"(dim={EMBEDDING_DIMENSIONS})")
    return index


def save_artifacts(df: pd.DataFrame, embeddings: np.ndarray,
                   index: faiss.IndexFlatIP):
    """Save all pipeline outputs."""
    print("[5/5] Saving artifacts...")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"    FAISS index → {FAISS_INDEX_PATH}")

    # Save raw embeddings
    np.save(str(EMBEDDINGS_NPY_PATH), embeddings)
    print(f"    Embeddings → {EMBEDDINGS_NPY_PATH}")

    # Save metadata as CSV
    df.to_csv(str(METADATA_PATH), index=False)
    print(f"    Metadata → {METADATA_PATH}")

    # Also save a lightweight JSON version for quick lookups
    catalog_json = DATA_DIR / "catalog.json"
    records = df.to_dict(orient="records")
    with open(catalog_json, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"    Catalog JSON → {catalog_json}")

    print("\n✅ Pipeline complete!")
    print(f"    Items indexed: {len(df)}")
    print(f"    Index size: {FAISS_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="StyleMind AI Data Pipeline")
    parser.add_argument("--csv", required=True, help="Path to styles.csv")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation (use existing)")
    args = parser.parse_args()

    df = load_and_clean(args.csv)
    embedding_strings = build_embedding_strings(df)

    if args.skip_embeddings and EMBEDDINGS_NPY_PATH.exists():
        print("[3/5] Loading existing embeddings...")
        embeddings = np.load(str(EMBEDDINGS_NPY_PATH))
    else:
        embeddings = generate_embeddings(embedding_strings)

    index = build_faiss_index(embeddings)
    save_artifacts(df, embeddings, index)


if __name__ == "__main__":
    main()
