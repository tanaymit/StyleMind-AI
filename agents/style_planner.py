"""
StyleMind AI - Style Planning Agent
Generates abstract outfit blueprints using chain-of-thought reasoning.

Uses the HEAVY tier model (Sonnet 4.6) — this requires creative reasoning.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from agents.llm_client import get_llm_client, LLMClient
from agents.intent_parser import ParsedIntent
from agents.taste_profile import TasteProfile


# ── Output Schema ──────────────────────────────────────────────────────────

class BlueprintItem(BaseModel):
    """A single item slot in an outfit blueprint."""
    slot: str = Field(description="E.g., 'top', 'bottom', 'shoes', 'outerwear', 'accessory'")
    article_type: str = Field(description="Specific type, e.g., 'oxford shirt', 'slim chinos'")
    target_color: str = Field(description="Desired color or color family")
    formality_note: str = Field(description="How this item contributes to the formality target")
    reasoning: str = Field(description="Chain-of-thought: why this item for this person/occasion")
    search_query: str = Field(
        description="Natural language query for catalog retrieval, "
                    "e.g., 'tailored navy chinos smart casual men'"
    )


class OutfitBlueprint(BaseModel):
    """A complete outfit blueprint (before catalog lookup)."""
    outfit_name: str = Field(description="Short evocative name, e.g., 'The Effortless Edge'")
    outfit_concept: str = Field(description="1-2 sentence concept statement")
    items: list[BlueprintItem] = Field(description="3-6 items composing the outfit")
    color_palette: list[str] = Field(description="The 2-4 colors this outfit is built around")
    social_signal: str = Field(
        description="What wearing this outfit communicates"
    )


class PlannerOutput(BaseModel):
    """Full output from the Style Planning Agent."""
    occasion_analysis: str = Field(description="Agent's interpretation of the occasion")
    profile_integration: str = Field(description="How the taste profile influenced the plan")
    blueprints: list[OutfitBlueprint] = Field(description="List of outfit blueprints (1–3 items)")


# ── System Prompt ──────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are the Style Planning Agent for StyleMind AI, a personal styling system.

Your job is to create ABSTRACT outfit blueprints - you decide WHAT kinds of items are needed, but you do NOT pick specific products. That happens later in the retrieval step.

You will receive:
1. A structured intent (parsed from the user's request)
2. The user's taste profile
3. REQUIRED_OUTFITS: the exact number of blueprints you must produce

You must output a valid JSON object matching the schema below. Do NOT include any text outside the JSON.

## Output Schema
{schema}

## CRITICAL COUNT RULE
The `blueprints` array MUST contain EXACTLY `REQUIRED_OUTFITS` items — no more, no fewer.
If REQUIRED_OUTFITS=1, output exactly 1 blueprint. If 2, exactly 2. If 3, exactly 3.
Finishing early or adding extras is an error.

## Chain-of-Thought Reasoning Rules
For EVERY item in every blueprint, you MUST provide explicit reasoning that covers:
1. Why this type of item fits the occasion
2. Why this color works with the palette
3. How it aligns (or deliberately departs from) the user's taste profile
4. What social signal it contributes to the overall outfit

## Style Rules
1. Every outfit must have at minimum: top, bottom, shoes. Accessories and outerwear are added when context demands.
2. Color palettes should be cohesive. Use the 60-30-10 rule: 60% dominant color, 30% secondary, 10% accent.
3. Formality must be consistent across all items in an outfit. Don't pair a blazer with flip-flops.
4. If the profile shows rejections, AVOID those items unless the user explicitly overrides.
5. If the user has preferred colors/styles, lean into them while keeping the outfit fresh.
6. Generate DIFFERENT concepts across blueprints - don't just swap one item.
7. Each search_query should be specific enough for good catalog retrieval: include article type, color, formality, and gender.

## Few-Shot Example

Intent: {{formality: "Smart Casual", occasion: "dinner", budget_max: 200, season: "Fall", gender_expression: "masculine"}}
Profile: Prefers minimalist style, likes navy/charcoal/white, avoids bright patterns, rejected chunky sneakers 2x

Output includes reasoning like:
"For a smart casual dinner, the foundation should be clean and slightly elevated. Given this user's minimalist preference and the Fall season, I'll anchor around navy and charcoal with clean silhouettes. No chunky sneakers per rejection history - instead, clean leather options."
"""


# ── JSON helpers ───────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _repair_truncated_json(text: str) -> str:
    """
    Best-effort repair of JSON truncated mid-string by the token limit.
    Keeps only the blueprints that parsed completely, closes the JSON object.
    """
    # Find the last successfully closed blueprint object
    # by scanning for '}' at depth=1 inside the blueprints array
    depth = 0
    last_complete_bp_end = -1
    in_string = False
    escape_next = False
    bp_array_start = text.find('"blueprints"')

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 1 and i > bp_array_start:
                last_complete_bp_end = i

    if last_complete_bp_end == -1:
        raise json.JSONDecodeError("Cannot repair", text, 0)

    # Trim to last complete blueprint and close the JSON
    repaired = text[:last_complete_bp_end + 1] + "]}"
    # Ensure occasion_analysis and profile_integration fields exist
    if '"occasion_analysis"' not in repaired:
        repaired = '{"occasion_analysis":"","profile_integration":"",' + repaired[1:]
    return repaired


# ── Planner Agent ──────────────────────────────────────────────────────────

class StylePlanner:
    """Generates outfit blueprints using chain-of-thought reasoning."""

    def __init__(self, client: Optional[LLMClient] = None):
        # Heavy tier — creative reasoning needs the stronger model
        self.client = client or get_llm_client(tier="heavy")

    def plan(self, intent: ParsedIntent,
             profile: Optional[TasteProfile] = None,
             num_outfits: int = 2) -> PlannerOutput:
        """Generate outfit blueprints from parsed intent and taste profile."""
        has_profile_data = profile and (
            profile.total_sessions > 0
            or profile.gender_expression not in ("unspecified", "", None)
            or bool(profile.color_preferences.preferred)
            or bool(profile.color_preferences.avoided)
        )
        if has_profile_data:
            profile_context = profile.get_profile_summary()
        else:
            profile_context = "(New user - no taste history. Generate broadly appealing options.)"

        schema_str = json.dumps(PlannerOutput.model_json_schema(), indent=2)
        system_prompt = PLANNER_SYSTEM_PROMPT.format(schema=schema_str)

        user_message = f"""REQUIRED_OUTFITS: {num_outfits}

## Parsed Intent
{intent.model_dump_json(indent=2)}

## User Taste Profile
{profile_context}

The `blueprints` array MUST have EXACTLY {num_outfits} item{"s" if num_outfits != 1 else ""}. Output ONLY valid JSON. No other text."""

        # Scale token budget: ~1400 tokens per outfit + 400 overhead
        max_tokens = min(400 + num_outfits * 1400, 4096)

        raw_output = self.client.complete(
            system=system_prompt,
            user=user_message,
            temperature=0.3,
            max_tokens=max_tokens,
            json_mode=True,
        )

        cleaned = _strip_fences(raw_output)

        try:
            parsed_data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt recovery: truncate to last complete top-level key boundary
            cleaned = _repair_truncated_json(cleaned)
            try:
                parsed_data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse planner output: {e}\n"
                    f"Raw output (first 500 chars): {raw_output[:500]}"
                ) from e

        try:
            output = PlannerOutput(**parsed_data)
        except Exception as e:
            raise ValueError(f"Planner schema validation failed: {e}") from e

        return output

    def plan_to_dict(self, intent: ParsedIntent,
                     profile: Optional[TasteProfile] = None) -> dict:
        """Plan and return as a plain dict."""
        return self.plan(intent, profile).model_dump()
