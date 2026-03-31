#!/usr/bin/env python3
"""
StyleMind AI - Quick Setup
Generates OpenAI embeddings and builds the FAISS index.

Usage:
    export OPENAI_API_KEY=sk-...
    python setup.py

Prerequisites:
    pip install -r requirements.txt
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

EMBEDDING_MODEL = "azure/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 500

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
PROFILES_DIR = PROJECT_ROOT / "profiles"

for d in [DATA_DIR, EMBEDDINGS_DIR, PROFILES_DIR]:
    d.mkdir(exist_ok=True)


def check_api_key():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["OPENAI_API_KEY"] = key
                    break
    if not key or key == "your-openai-api-key-here":
        print("ERROR: No OpenAI API key found.")
        print("Set it via: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    return key


def check_data():
    csv_path = DATA_DIR / "catalog_cleaned.csv"
    meta_path = EMBEDDINGS_DIR / "catalog_metadata.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif meta_path.exists():
        return pd.read_csv(meta_path)
    else:
        raw = PROJECT_ROOT / "styles.csv"
        if not raw.exists():
            raw = DATA_DIR / "styles.csv"
        if not raw.exists():
            print("ERROR: No dataset found. Place styles.csv in project root or data/.")
            sys.exit(1)
        print("Found raw styles.csv, running cleaning pipeline...")
        from data_pipeline import load_and_clean
        df = load_and_clean(str(raw))
        df.to_csv(csv_path, index=False)
        df.to_csv(meta_path, index=False)
        return df


def generate_embeddings(df, api_key):
    cache_path = EMBEDDINGS_DIR / "catalog_embeddings.npy"
    if cache_path.exists():
        embeddings = np.load(str(cache_path))
        if len(embeddings) == len(df):
            print(f"Using cached embeddings: {embeddings.shape}")
            return embeddings

    if "embedding_string" in df.columns:
        texts = df["embedding_string"].tolist()
    else:
        texts = (
            df["productDisplayName"] + " | " + df["articleType"] + " | " +
            df["usage"] + " | " + df["baseColour"] + " | " + df["season"] + " | " +
            df["gender"]
        ).tolist()

    print(f"Generating embeddings for {len(texts)} items via {EMBEDDING_MODEL}...")
    client = OpenAI(api_key=api_key, base_url="https://ai-gateway.andrew.cmu.edu")
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches}...", end=" ", flush=True)
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                all_embeddings.extend([e.embedding for e in resp.data])
                print("done")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"retry ({e})...")
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"FATAL: {e}")
                    if all_embeddings:
                        partial = np.array(all_embeddings, dtype=np.float32)
                        np.save(str(EMBEDDINGS_DIR / "catalog_embeddings_partial.npy"), partial)
                    sys.exit(1)
        if batch_num < total_batches:
            time.sleep(0.3)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    np.save(str(cache_path), embeddings)
    print(f"  Saved: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings):
    print("Building FAISS index...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(normalized)
    index_path = EMBEDDINGS_DIR / "catalog.index"
    faiss.write_index(index, str(index_path))
    print(f"  {index.ntotal} vectors, {index_path.stat().st_size / 1024 / 1024:.1f} MB")


def verify():
    print("\nVerifying...")
    index = faiss.read_index(str(EMBEDDINGS_DIR / "catalog.index"))
    meta = pd.read_csv(EMBEDDINGS_DIR / "catalog_metadata.csv")
    emb = np.load(str(EMBEDDINGS_DIR / "catalog_embeddings.npy"))
    assert index.ntotal == len(meta) == len(emb)
    q = emb[0:1] / np.linalg.norm(emb[0:1])
    scores, ids = index.search(q, 3)
    print(f"  Test search: {meta.iloc[ids[0][0]]['productDisplayName']} (score={scores[0][0]:.3f})")
    print("\nSetup complete!")


def main():
    print("=" * 60)
    print("StyleMind AI - Setup (Embeddings + FAISS)")
    print("=" * 60 + "\n")
    api_key = check_api_key()
    df = check_data()
    print(f"Dataset: {len(df)} items\n")
    embeddings = generate_embeddings(df, api_key)
    build_faiss_index(embeddings)
    meta_path = EMBEDDINGS_DIR / "catalog_metadata.csv"
    if not meta_path.exists():
        df.to_csv(meta_path, index=False)
    verify()


if __name__ == "__main__":
    main()
