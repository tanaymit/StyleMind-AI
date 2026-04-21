"""
StyleMind AI - Lookbook Generator
Generates 120–180 word aspirational outfit prose using the heavy-tier model.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from agents.llm_client import get_llm_client, LLMClient
from agents.outfit_assembler import ScoredOutfit
from agents.intent_parser import ParsedIntent
from agents.taste_profile import TasteProfile


# ── System Prompt ──────────────────────────────────────────────────────────

LOOKBOOK_SYSTEM = """You are the Lookbook Writer for StyleMind AI — a personal styling assistant.

Your job: write exactly 120–180 words of aspirational fashion prose for a specific outfit recommendation.

The prose must:
- Describe how the outfit FEELS to wear and how others will read it
- Reference items by their role, not brand name ("the slim chinos", "the leather oxford")
- Connect the outfit emotionally to the occasion
- Use sensory and atmospheric language (texture, light, movement)
- Reflect the user's style identity where naturally relevant
- Close with one sharp, memorable sentence about the social signal

Rules:
- No lists, no headers, no markdown
- No generic phrases like "this outfit is perfect for" or "you'll look great"
- No brand names
- Output ONLY the prose. Nothing else.
- Stay between 120 and 180 words — count carefully."""


# ── Output Schema ──────────────────────────────────────────────────────────

class LookbookEntry(BaseModel):
    outfit_name: str
    prose: str
    word_count: int


# ── Generator ─────────────────────────────────────────────────────────────

class LookbookGenerator:
    """Generates aspirational lookbook prose for assembled outfits."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_llm_client(tier="heavy")

    def generate(
        self,
        outfit: ScoredOutfit,
        intent: ParsedIntent,
        profile: Optional[TasteProfile] = None,
    ) -> LookbookEntry:
        profile_note = ""
        if profile and profile.total_sessions > 0:
            profile_note = f"\nUser's style identity: {profile.style_identity}"
            if profile.style_archetypes:
                profile_note += f"\nStyle archetypes: {', '.join(profile.style_archetypes)}"

        items_desc = "\n".join(
            f"  • {item.slot.upper()}: {item.product_name} in {item.color} — ${item.price:.0f}"
            for item in outfit.items
        )

        user_msg = f"""Outfit name: {outfit.blueprint_name}
Concept: {outfit.blueprint_concept}
Occasion: {intent.occasion} ({intent.formality_level})
Social signal: {outfit.social_signal}
Color palette: {', '.join(outfit.color_palette)}

Items:
{items_desc}
{profile_note}

Write the lookbook prose now. Remember: 120–180 words, atmospheric, no lists."""

        prose = self.client.complete(
            system=LOOKBOOK_SYSTEM,
            user=user_msg,
            temperature=0.75,
            max_tokens=350,
        ).strip()

        return LookbookEntry(
            outfit_name=outfit.blueprint_name,
            prose=prose,
            word_count=len(prose.split()),
        )

    def generate_batch(
        self,
        outfits: list[ScoredOutfit],
        intent: ParsedIntent,
        profile: Optional[TasteProfile] = None,
    ) -> list[LookbookEntry]:
        """Generate lookbooks for multiple outfits — call in ThreadPoolExecutor for speed."""
        return [self.generate(o, intent, profile) for o in outfits]
