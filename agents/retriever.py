"""
StyleMind AI - Product Retriever
Hybrid retrieval: FAISS semantic search + structured metadata filters.
"""

from __future__ import annotations

from typing import Optional
import os
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    OPENAI_API_KEY,
    TOP_K_RETRIEVAL,
    USAGE_FORMALITY_MAP,
)

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://ai-gateway.andrew.cmu.edu")

# ── Slot → catalog category mapping ──────────────────────────────────────────
# Maps blueprint slot names to allowed masterCategory / subCategory / articleType.
# This prevents e.g. wallets appearing as outerwear, or t-shirts appearing in winter.

_SLOT_RULES: dict[str, dict] = {
    "top": {
        "masterCategory": ["Apparel"],
        "subCategory": ["Topwear"],
        # Outerwear article types are handled by the "outerwear" slot
        "blocked_articleType_keywords": ["jacket", "blazer", "waistcoat", "cardigan", "coat"],
    },
    "bottom": {
        "masterCategory": ["Apparel"],
        "subCategory": ["Bottomwear", "Dress"],
    },
    "shoes": {
        "masterCategory": ["Footwear"],
    },
    "outerwear": {
        "masterCategory": ["Apparel"],
        "subCategory": ["Topwear"],
        # Only these article types qualify as outerwear
        "required_articleType_keywords": [
            "jacket", "blazer", "waistcoat", "cardigan",
            "sweater", "sweatshirt", "hoodie", "coat", "rain jacket",
        ],
    },
    "accessory": {
        "masterCategory": ["Accessories"],
        # Wallets and bags are not general accessories for an outfit
        "blocked_subCategory": ["Wallets", "Bags"],
    },
    "bag": {
        "masterCategory": ["Accessories"],
        "subCategory": ["Bags"],
    },
    "dress": {
        "masterCategory": ["Apparel"],
        "subCategory": ["Dress"],
    },
    "innerwear": {
        "masterCategory": ["Apparel"],
        "subCategory": ["Innerwear"],
    },
}

# ── Color family groupings for harmony scoring ────────────────────────────────
_COLOR_FAMILIES: dict[str, list[str]] = {
    "white_light":   ["white", "cream", "ivory", "off-white", "light", "oatmeal"],
    "neutral_warm":  ["beige", "camel", "tan", "khaki", "stone", "sand", "taupe"],
    "neutral_cool":  ["grey", "gray", "silver", "slate"],
    "dark_neutral":  ["black", "charcoal", "dark grey", "dark gray"],
    "navy_indigo":   ["navy", "indigo", "cobalt", "dark blue"],
    "blue":          ["blue", "denim", "chambray", "sky blue", "powder blue"],
    "earth_green":   ["olive", "green", "sage", "forest", "hunter", "khaki green"],
    "warm_earth":    ["brown", "rust", "copper", "terracotta", "cognac", "tobacco"],
    "red_wine":      ["red", "burgundy", "maroon", "wine", "oxblood", "crimson", "bordeaux"],
    "pink_blush":    ["pink", "blush", "rose", "coral", "salmon", "mauve"],
    "purple":        ["purple", "lavender", "violet", "plum", "lilac"],
    "yellow_gold":   ["yellow", "mustard", "gold", "amber", "saffron"],
    "teal_mint":     ["teal", "mint", "turquoise", "aqua", "cyan"],
}

# Which family pairs are compatible (score boost), clashing (penalty), or neutral
_FAMILY_COMPAT: dict[frozenset, float] = {
    # Neutrals work with everything → no special entry needed (default 0.5)
    # High compatibility pairs
    frozenset({"neutral_warm", "dark_neutral"}):  0.9,
    frozenset({"neutral_warm", "navy_indigo"}):   0.9,
    frozenset({"neutral_warm", "warm_earth"}):    1.0,
    frozenset({"neutral_warm", "red_wine"}):      0.8,
    frozenset({"neutral_cool", "dark_neutral"}):  0.9,
    frozenset({"neutral_cool", "navy_indigo"}):   0.9,
    frozenset({"dark_neutral", "navy_indigo"}):   0.8,
    frozenset({"dark_neutral", "red_wine"}):      0.8,
    frozenset({"dark_neutral", "earth_green"}):   0.8,
    frozenset({"navy_indigo", "white_light"}):    0.95,
    frozenset({"navy_indigo", "neutral_warm"}):   0.9,
    frozenset({"earth_green", "neutral_warm"}):   0.9,
    frozenset({"earth_green", "warm_earth"}):     0.9,
    frozenset({"warm_earth", "dark_neutral"}):    0.85,
    frozenset({"red_wine", "dark_neutral"}):      0.85,
    frozenset({"white_light", "dark_neutral"}):   0.9,
    frozenset({"white_light", "navy_indigo"}):    0.95,
    frozenset({"blue", "neutral_cool"}):          0.8,
    # Clashing pairs (lower than default 0.5)
    frozenset({"red_wine", "pink_blush"}):        0.2,
    frozenset({"red_wine", "yellow_gold"}):       0.2,
    frozenset({"pink_blush", "yellow_gold"}):     0.2,
    frozenset({"purple", "yellow_gold"}):         0.3,
    frozenset({"purple", "red_wine"}):            0.3,
    frozenset({"teal_mint", "pink_blush"}):       0.3,
}


def _color_family(color_name: str) -> Optional[str]:
    """Return the color family key for a given color name, or None."""
    cl = color_name.lower().strip()
    for family, members in _COLOR_FAMILIES.items():
        if any(m in cl or cl in m for m in members):
            return family
    return None


def _family_compat_score(family_a: Optional[str], family_b: Optional[str]) -> float:
    """Return [0,1] compatibility between two color families."""
    if family_a is None or family_b is None:
        return 0.5
    if family_a == family_b:
        return 1.0
    # Anything involving a neutral family is naturally compatible
    neutral = {"white_light", "neutral_warm", "neutral_cool", "dark_neutral"}
    if family_a in neutral or family_b in neutral:
        return 0.75
    return _FAMILY_COMPAT.get(frozenset({family_a, family_b}), 0.5)


def palette_harmony_score(item_color: str, palette: list[str]) -> float:
    """Score how well an item's color fits a palette using family compatibility."""
    if not palette:
        return 0.5
    item_family = _color_family(item_color)
    scores = [_family_compat_score(item_family, _color_family(p)) for p in palette]
    return max(scores)  # best match against any palette color


class ProductRetriever:
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
                    f"FAISS index not found at {FAISS_INDEX_PATH}. Run setup.py first."
                )
            self._index = faiss.read_index(str(FAISS_INDEX_PATH))
        return self._index

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            if not METADATA_PATH.exists():
                raise FileNotFoundError(
                    f"Metadata not found at {METADATA_PATH}. Run setup.py first."
                )
            self._metadata = pd.read_csv(str(METADATA_PATH))
        return self._metadata

    def _embed_query(self, query: str) -> np.ndarray:
        response = self._embedding_client.embeddings.create(
            model=EMBEDDING_MODEL, input=[query],
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
        slot: Optional[str] = None,
    ) -> list[dict]:
        fetch_k = min(top_k * 10, self.index.ntotal)
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, fetch_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = self.metadata.iloc[idx].to_dict()
            row["similarity_score"] = float(score)
            candidates.append(row)

        # ── Fallback cascade: NEVER relax season or gender ────────────────
        # Each step relaxes one non-essential constraint.
        filter_passes = [
            # Full filters
            dict(gender=gender, formality=formality, season=season,
                 color=color, budget_max=budget_max,
                 excluded_colors=excluded_colors, slot=slot),
            # Drop color hint (keep season + gender + budget)
            dict(gender=gender, formality=formality, season=season,
                 color=None, budget_max=budget_max,
                 excluded_colors=excluded_colors, slot=slot),
            # Drop formality + color (keep season + gender + budget)
            dict(gender=gender, formality=None, season=season,
                 color=None, budget_max=budget_max,
                 excluded_colors=excluded_colors, slot=slot),
            # Drop formality + color + budget (keep season + gender)
            dict(gender=gender, formality=None, season=season,
                 color=None, budget_max=None,
                 excluded_colors=None, slot=slot),
            # Drop formality + color + budget + excluded_colors (keep season + gender)
            dict(gender=gender, formality=None, season=season,
                 color=None, budget_max=None,
                 excluded_colors=None, slot=slot),
            # Last resort: season + slot only (no gender — broad fallback)
            dict(gender=None, formality=None, season=season,
                 color=None, budget_max=None,
                 excluded_colors=None, slot=slot),
        ]

        filtered = []
        for pass_kwargs in filter_passes:
            filtered = self._apply_filters(candidates, **pass_kwargs)
            if len(filtered) >= 3:
                break

        # Absolute last resort: slot category only
        if len(filtered) < 3:
            filtered = self._apply_filters(candidates, slot=slot) or candidates

        return filtered[:top_k]

    def _apply_filters(
        self,
        candidates: list[dict],
        gender: Optional[str] = None,
        formality: Optional[str] = None,
        season: Optional[str] = None,
        color: Optional[str] = None,
        budget_max: Optional[float] = None,
        excluded_colors: Optional[list[str]] = None,
        slot: Optional[str] = None,
    ) -> list[dict]:
        filtered = list(candidates)

        # ── Slot-based category hard filter (highest priority) ────────────
        if slot:
            slot_lower = slot.lower().strip()
            rules = _SLOT_RULES.get(slot_lower)
            if rules:
                # masterCategory allow-list
                if "masterCategory" in rules:
                    allowed_mc = rules["masterCategory"]
                    filtered = [c for c in filtered if c.get("masterCategory") in allowed_mc]

                # subCategory allow-list (if specified)
                if "subCategory" in rules:
                    allowed_sc = rules["subCategory"]
                    filtered = [c for c in filtered if c.get("subCategory") in allowed_sc]

                # blocked subCategories
                if "blocked_subCategory" in rules:
                    blocked_sc = rules["blocked_subCategory"]
                    filtered = [c for c in filtered if c.get("subCategory") not in blocked_sc]

                # required articleType keywords (outerwear case)
                if "required_articleType_keywords" in rules:
                    kws = rules["required_articleType_keywords"]
                    filtered = [
                        c for c in filtered
                        if any(kw in c.get("articleType", "").lower() for kw in kws)
                    ]

                # blocked articleType keywords (top case — exclude outerwear)
                if "blocked_articleType_keywords" in rules:
                    kws = rules["blocked_articleType_keywords"]
                    filtered = [
                        c for c in filtered
                        if not any(kw in c.get("articleType", "").lower() for kw in kws)
                    ]

        # ── Gender ────────────────────────────────────────────────────────
        if gender and gender.lower() not in ("unspecified", ""):
            gender_map = {
                "masculine": ["Men", "Unisex"],
                "feminine":  ["Women", "Unisex", "Girls"],
                "androgynous": ["Unisex", "Men", "Women"],
                "men":   ["Men", "Unisex"],
                "women": ["Women", "Unisex"],
            }
            allowed = gender_map.get(gender.lower(), ["Men", "Women", "Unisex"])
            filtered = [c for c in filtered if c.get("gender") in allowed]

        # ── Formality ─────────────────────────────────────────────────────
        if formality:
            allowed_usage = USAGE_FORMALITY_MAP.get(formality, [])
            if allowed_usage:
                filtered = [c for c in filtered if c.get("usage") in allowed_usage]

        # ── Season ────────────────────────────────────────────────────────
        if season:
            season_map = {
                "Spring": "Spring", "Summer": "Summer",
                "Fall": "Fall", "Autumn": "Fall", "Winter": "Winter",
            }
            mapped = season_map.get(season, season)
            filtered = [c for c in filtered
                        if c.get("season") in (mapped, "All Season")]

        # ── Target color (soft boost — sort matching colors first) ────────
        if color:
            color_lower = color.lower()
            matching = [c for c in filtered
                        if color_lower in c.get("baseColour", "").lower()
                        or c.get("baseColour", "").lower() in color_lower]
            non_matching = [c for c in filtered if c not in matching]
            filtered = matching + non_matching

        # ── Budget ────────────────────────────────────────────────────────
        if budget_max:
            filtered = [c for c in filtered if c.get("price", 0) <= budget_max]

        # ── Excluded colors ───────────────────────────────────────────────
        if excluded_colors:
            excluded_lower = {ec.lower() for ec in excluded_colors}
            filtered = [c for c in filtered
                        if c.get("baseColour", "").lower() not in excluded_lower]

        return filtered

    def retrieve_for_blueprint_slot(
        self,
        search_query: str,
        slot: str,
        target_color: str,
        intent_data: dict,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict]:
        return self.retrieve(
            query=search_query,
            top_k=top_k,
            gender=intent_data.get("gender_expression"),
            formality=intent_data.get("formality_level"),
            season=intent_data.get("season"),
            color=target_color if target_color.lower() != "any" else None,
            budget_max=intent_data.get("budget_max"),
            excluded_colors=intent_data.get("color_avoidances", []),
            slot=slot,
        )
