"""
StyleMind AI - Outfit Assembly & Scoring
Combines per-slot product candidates into complete outfits and scores them.
Pure rule-based — no LLM call required, so this is instant.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from agents.style_planner import OutfitBlueprint
from agents.retriever import palette_harmony_score


# ── Output Schemas ─────────────────────────────────────────────────────────

class AssembledItem(BaseModel):
    slot: str
    article_type: str
    product_name: str
    color: str
    price: float
    similarity_score: float
    product_id: Optional[str] = None


class ScoredOutfit(BaseModel):
    blueprint_name: str
    blueprint_concept: str
    color_palette: list[str]
    social_signal: str
    items: list[AssembledItem]
    total_price: float
    budget_score: float        # 0–1, 1.0 = all items within budget
    color_harmony_score: float  # 0–1
    avg_similarity: float
    overall_score: float        # weighted composite
    lookbook_prose: Optional[str] = None  # filled by LookbookGenerator


# ── Assembler ──────────────────────────────────────────────────────────────

class OutfitAssembler:
    """
    Selects one product per blueprint slot and scores the resulting outfit.

    slot_candidates: {slot_name: [retrieved_product_dict, ...]}
    Each product dict must have: productDisplayName, baseColour, price,
    similarity_score, id.
    """

    def assemble(
        self,
        blueprint: OutfitBlueprint,
        slot_candidates: dict[str, list[dict]],
        budget_max: Optional[float] = None,
    ) -> ScoredOutfit:
        items: list[AssembledItem] = []

        # Distribute total budget across slots so per-item cap is proportional.
        # 1.5× headroom accounts for natural price variation between slot types.
        num_slots = max(len(blueprint.items), 1)
        per_item_budget = (budget_max / num_slots * 1.5) if budget_max else None

        for bp_item in blueprint.items:
            candidates = slot_candidates.get(bp_item.slot, [])
            if not candidates:
                continue
            best = self._pick_best(candidates, budget_max=per_item_budget,
                                   palette=blueprint.color_palette)
            if best:
                items.append(AssembledItem(
                    slot=bp_item.slot,
                    article_type=bp_item.article_type,
                    product_name=best.get("productDisplayName", "Unknown"),
                    color=best.get("baseColour", ""),
                    price=float(best.get("price", 0)),
                    similarity_score=float(best.get("similarity_score", 0)),
                    product_id=str(best.get("id", "")),
                ))

        total_price = sum(i.price for i in items)
        budget_score = self._budget_score(total_price, budget_max)
        color_harmony = self._color_harmony(items, blueprint.color_palette)
        avg_sim = sum(i.similarity_score for i in items) / max(len(items), 1)

        # Weighted composite: similarity matters most, then color harmony, then budget
        overall = (0.5 * avg_sim) + (0.3 * color_harmony) + (0.2 * budget_score)

        return ScoredOutfit(
            blueprint_name=blueprint.outfit_name,
            blueprint_concept=blueprint.outfit_concept,
            color_palette=blueprint.color_palette,
            social_signal=blueprint.social_signal,
            items=items,
            total_price=total_price,
            budget_score=budget_score,
            color_harmony_score=color_harmony,
            avg_similarity=avg_sim,
            overall_score=min(overall, 1.0),
        )

    def assemble_all(
        self,
        blueprints: list[OutfitBlueprint],
        all_slot_candidates: dict[str, dict[str, list[dict]]],
        budget_max: Optional[float] = None,
    ) -> list[ScoredOutfit]:
        """Assemble all blueprints, sorted by overall_score descending."""
        outfits = [
            self.assemble(bp, all_slot_candidates.get(bp.outfit_name, {}), budget_max)
            for bp in blueprints
        ]
        return sorted(outfits, key=lambda o: o.overall_score, reverse=True)

    # ── Private helpers ──────────────────────────────────────────────────

    def _pick_best(
        self,
        candidates: list[dict],
        budget_max: Optional[float],
        palette: Optional[list[str]] = None,
    ) -> Optional[dict]:
        if not candidates:
            return None
        within = [c for c in candidates
                  if budget_max is None or float(c.get("price", 0)) <= budget_max]
        pool = within if within else candidates

        def _score(c: dict) -> float:
            sim = float(c.get("similarity_score", 0))
            if palette:
                color_fit = palette_harmony_score(c.get("baseColour", ""), palette)
                # 70% semantic similarity, 30% color family fit
                return 0.70 * sim + 0.30 * color_fit
            return sim

        return max(pool, key=_score)

    def _budget_score(self, total_price: float, budget_max: Optional[float]) -> float:
        if budget_max is None or budget_max <= 0:
            return 1.0
        if total_price <= budget_max:
            return 1.0
        over_ratio = (total_price - budget_max) / budget_max
        return max(0.0, 1.0 - over_ratio)

    def _color_harmony(
        self,
        items: list[AssembledItem],
        palette: list[str],
    ) -> float:
        """Score overall outfit color harmony using color family compatibility."""
        if not items or not palette:
            return 0.5
        scores = [palette_harmony_score(item.color, palette) for item in items]
        return sum(scores) / len(scores)
