"""
StyleMind AI - Data Verification
Verifies the cleaned dataset and embedding strings are correct.
No external dependencies needed beyond pandas/numpy.

Run: python tests/test_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def test_cleaned_dataset():
    """Verify the cleaned dataset."""
    print("Testing cleaned dataset...")

    csv_path = Path(__file__).parent.parent / "data" / "catalog_cleaned.csv"
    assert csv_path.exists(), f"Cleaned CSV not found at {csv_path}"

    df = pd.read_csv(csv_path)

    # Size check
    assert len(df) > 40000, f"Expected 40K+ items, got {len(df)}"
    print(f"  Items: {len(df)}")

    # Required columns
    required = ["id", "gender", "masterCategory", "subCategory", "articleType",
                 "baseColour", "season", "year", "usage", "productDisplayName",
                 "price", "embedding_string"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    print(f"  Columns: ✓ all {len(required)} required columns present")

    # No critical nulls
    critical = ["productDisplayName", "articleType", "gender", "masterCategory"]
    for col in critical:
        nulls = df[col].isnull().sum()
        assert nulls == 0, f"Found {nulls} nulls in {col}"
    print("  Nulls: ✓ no nulls in critical columns")

    # Only relevant categories
    cats = set(df["masterCategory"].unique())
    assert cats <= {"Apparel", "Accessories", "Footwear"}, f"Unexpected categories: {cats}"
    print(f"  Categories: {cats}")

    # Prices are reasonable
    assert df["price"].min() >= 5, f"Min price too low: {df['price'].min()}"
    assert df["price"].max() <= 350, f"Max price too high: {df['price'].max()}"
    print(f"  Prices: ${df['price'].min():.2f} - ${df['price'].max():.2f} (mean ${df['price'].mean():.2f})")

    # Embedding strings are well-formed
    sample = df["embedding_string"].iloc[0]
    parts = sample.split(" | ")
    assert len(parts) == 6, f"Expected 6 parts in embedding string, got {len(parts)}: {sample}"
    print(f"  Embedding string format: ✓ ({len(parts)} fields)")
    print(f"  Sample: '{sample[:80]}...'")

    # Gender distribution
    genders = df["gender"].value_counts().to_dict()
    assert "Men" in genders and "Women" in genders
    print(f"  Gender split: Men={genders.get('Men',0)}, Women={genders.get('Women',0)}, "
          f"Unisex={genders.get('Unisex',0)}")

    print("  ✅ Dataset verification passed\n")
    return df


def test_embedding_strings():
    """Verify embedding strings file."""
    print("Testing embedding strings file...")

    path = Path(__file__).parent.parent / "data" / "embedding_strings.csv"
    assert path.exists(), f"Embedding strings not found at {path}"

    strings = pd.read_csv(path)
    assert len(strings) > 40000
    print(f"  Count: {len(strings)}")
    print("  ✅ Embedding strings file verified\n")


def test_metadata_for_retriever():
    """Verify metadata file in embeddings dir (used by retriever)."""
    print("Testing retriever metadata...")

    path = Path(__file__).parent.parent / "embeddings" / "catalog_metadata.csv"
    assert path.exists(), f"Metadata not found at {path}"

    df = pd.read_csv(path)
    assert len(df) > 40000
    assert "price" in df.columns
    assert "articleType" in df.columns
    print(f"  Items: {len(df)}, Columns: {len(df.columns)}")
    print("  ✅ Retriever metadata verified\n")


def test_faiss_index():
    """Check if FAISS index exists (only if embeddings have been generated)."""
    print("Testing FAISS index...")

    index_path = Path(__file__).parent.parent / "embeddings" / "catalog.index"
    emb_path = Path(__file__).parent.parent / "embeddings" / "catalog_embeddings.npy"

    if not index_path.exists():
        print("  ⏭️  FAISS index not built yet (run setup.py with API key)")
        print("  This is expected before running the embedding pipeline.\n")
        return

    try:
        import faiss
        index = faiss.read_index(str(index_path))
        print(f"  Vectors: {index.ntotal}, Dimensions: {index.d}")

        embeddings = np.load(str(emb_path))
        assert index.ntotal == len(embeddings), "Index/embedding count mismatch"
        print(f"  Embeddings shape: {embeddings.shape}")
        print("  ✅ FAISS index verified\n")
    except ImportError:
        print("  ⏭️  faiss-cpu not installed, skipping index verification\n")


if __name__ == "__main__":
    print("=" * 60)
    print("StyleMind AI - Data Verification")
    print("=" * 60 + "\n")

    test_cleaned_dataset()
    test_embedding_strings()
    test_metadata_for_retriever()
    test_faiss_index()

    print("=" * 60)
    print("All data tests passed! ✅")
    print("=" * 60)
