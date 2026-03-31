"""
StyleMind AI - Product Retriever
Hybrid retrieval: FAISS semantic search + structured metadata filters.

Embeddings are always generated via OpenAI (text-embedding-3-small),
regardless of which LLM provider is used for generation.
"""

from __future__ import annotations

from typing import Optional
import os
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    OPENAI_API_KEY,
    TOP_K_RETRIEVAL,
    USAGE_FORMALITY_MAP,
)

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://ai-gateway.andrew.cmu.edu")

class ProductRetriever:
    """
    Retrieves products from the FAISS index using semantic search
    with optional structured metadata filters.

    Uses OpenAI embeddings for query encoding (same model used to build the index).
    """

    def __init__(self, api_key: str = OPENAI_API_KEY):
        from config import EMBEDDING_BASE_URL
        self._embedding_client = OpenAI(api_key=api_key, base_url=EMBEDDING_BASE_URL)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: Optional[pd.DataFrame] = None

    @property
    def index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            if not FAISS_INDEX_PATH.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {FAISS_INDEX_PATH}. "
                    "Run setup.py first."
                )
            self._index = faiss.read_index(str(FAISS_INDEX_PATH))
        return self._index

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            if not METADATA_PATH.exists():
                raise FileNotFoundError(
                    f"Metadata not found at {METADATA_PATH}. "
                    "Run setup.py first."
                )
            self._metadata = pd.read_csv(str(METADATA_PATH))
        return self._metadata

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string via OpenAI."""
        response = self._embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.reshape(1, -1)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        gender: Optional[str] = None,
        formality: Optional[str] = None,
        season: Optional[str] = None,
        color: Optional[str] = None,
        budget_max: Optional[float] = None,
        excluded_colors: Optional[list[str]] = None,
        article_type_hint: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve candidate products using semantic search + filters."""
        fetch_k = min(top_k * 5, self.index.ntotal)
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, fetch_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata.iloc[idx].to_dict()
            row["similarity_score"] = float(score)
            candidates.append(row)

        filtered = self._apply_filters(
            candidates, gender=gender, formality=formality, season=season,
            color=color, budget_max=budget_max, excluded_colors=excluded_colors,
            article_type_hint=article_type_hint,
        )

        # Relax filters if too aggressive
        if len(filtered) < 3:
            filtered = self._apply_filters(
                candidates, gender=gender, formality=None, season=None,
                color=color, budget_max=budget_max,
                excluded_colors=excluded_colors,
                article_type_hint=article_type_hint,
            )

        if len(filtered) < 3:
            filtered = candidates[:top_k]

        return filtered[:top_k]

    def _apply_filters(
        self, candidates: list[dict],
        gender: Optional[str] = None, formality: Optional[str] = None,
        season: Optional[str] = None, color: Optional[str] = None,
        budget_max: Optional[float] = None,
        excluded_colors: Optional[list[str]] = None,
        article_type_hint: Optional[str] = None,
    ) -> list[dict]:
        """Apply structured metadata filters to candidates."""
        filtered = candidates

        if gender and gender.lower() != "unspecified":
            gender_map = {
                "masculine": ["Men", "Unisex"],
                "feminine": ["Women", "Unisex"],
                "androgynous": ["Unisex", "Men", "Women"],
                "men": ["Men", "Unisex"],
                "women": ["Women", "Unisex"],
            }
            allowed = gender_map.get(gender.lower(), ["Men", "Women", "Unisex"])
            filtered = [c for c in filtered if c.get("gender") in allowed]

        if formality:
            allowed_usage = USAGE_FORMALITY_MAP.get(formality, [])
            if allowed_usage:
                filtered = [c for c in filtered if c.get("usage") in allowed_usage]

        if season:
            season_map = {"Spring": "Spring", "Summer": "Summer",
                          "Fall": "Fall", "Autumn": "Fall", "Winter": "Winter"}
            mapped = season_map.get(season, season)
            filtered = [c for c in filtered
                        if c.get("season") in [mapped, "All Season"]]

        if budget_max:
            filtered = [c for c in filtered if c.get("price", 0) <= budget_max]

        if excluded_colors:
            excluded_lower = [ec.lower() for ec in excluded_colors]
            filtered = [c for c in filtered
                        if c.get("baseColour", "").lower() not in excluded_lower]

        if article_type_hint:
            hint_lower = article_type_hint.lower()
            matching = [c for c in filtered
                        if hint_lower in c.get("articleType", "").lower()]
            non_matching = [c for c in filtered
                           if hint_lower not in c.get("articleType", "").lower()]
            filtered = matching + non_matching

        return filtered

    def retrieve_for_blueprint_slot(
        self, search_query: str, slot: str, target_color: str,
        intent_data: dict, top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict]:
        """Convenience: retrieve products for a specific blueprint slot."""
        return self.retrieve(
            query=search_query, top_k=top_k,
            gender=intent_data.get("gender_expression"),
            formality=intent_data.get("formality_level"),
            season=intent_data.get("season"),
            color=target_color if target_color.lower() != "any" else None,
            budget_max=intent_data.get("budget_max"),
            excluded_colors=intent_data.get("color_avoidances", []),
        )
