"""
StyleMind AI - Intent & Constraint Parser Agent
Parses natural language fashion requests into structured JSON constraints,
reconciling with the user's existing Taste Profile.

Uses the LIGHT tier model (Haiku 4.5) — this is a structured extraction task.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from agents.llm_client import get_llm_client, LLMClient
from agents.taste_profile import TasteProfile


# ── Output Schema ──────────────────────────────────────────────────────────

class ParsedIntent(BaseModel):
    """Structured output from the Intent & Constraint Parser."""
    occasion: str = Field(description="Brief description of the occasion")
    formality_level: str = Field(
        description="One of: Very Casual, Casual, Smart Casual, Semi-Formal, Formal, Black Tie"
    )
    social_context: Optional[str] = Field(default=None, description="E.g., 'first date'")

    budget_min: Optional[float] = Field(default=None, description="Min budget in USD")
    budget_max: Optional[float] = Field(default=None, description="Max budget in USD")
    climate: Optional[str] = Field(default=None, description="E.g., 'cold and rainy'")
    location: Optional[str] = Field(default=None, description="City or region")
    season: Optional[str] = Field(default=None, description="Spring, Summer, Fall, Winter")

    gender_expression: str = Field(default="unspecified")
    color_requests: list[str] = Field(default_factory=list)
    color_avoidances: list[str] = Field(default_factory=list)
    style_keywords: list[str] = Field(default_factory=list)

    profile_conflicts: list[str] = Field(default_factory=list)
    profile_enhancements: list[str] = Field(default_factory=list)

    num_outfits_requested: int = Field(default=2)
    special_instructions: Optional[str] = Field(default=None)


# ── System Prompt ──────────────────────────────────────────────────────────

PARSER_SYSTEM_PROMPT = """You are the Intent & Constraint Parser for StyleMind AI, a personal styling system.

Your job is to take a natural language fashion request and produce a structured JSON object that downstream agents can use to generate outfit recommendations.

You will be given:
1. The user's request (natural language)
2. The user's Taste Profile (may be empty for new users)

Your output must be a valid JSON object matching the schema below. Do NOT include any text outside the JSON.

## Output Schema
{schema}

## Rules
1. ALWAYS infer formality_level even if not stated explicitly. Use context clues from the occasion.
2. If the user mentions a budget, parse it into budget_min and budget_max. "Around $200" → budget_max: 200, budget_min: 150 (allow ~25% flex below).
3. If the user mentions a location, infer the climate and season if not stated. Season must be one of: Spring, Summer, Fall, Winter.
4. CHECK the Taste Profile for conflicts. If the user asks for something they've previously rejected, note it in profile_conflicts.
5. SEASON-AWARE PROFILE ENHANCEMENT: Only apply profile preferences that are appropriate for the CURRENT REQUEST'S season. If the user is asking for a Summer outfit, do NOT apply dark heavy colors or layering preferences from past Winter sessions. If the user is asking for a Winter outfit, do NOT suggest lightweight fabrics from Summer sessions. Color preferences are generally transferable, but fabric/weight/layering preferences are NOT cross-season.
6. GENDER: If the user explicitly states their gender or preferred style direction, use it. If the profile has a gender_expression set, inherit it. Otherwise use "unspecified". NEVER mix masculine and feminine gender expressions — once gender is known, be consistent.
7. Default to 2-3 outfits unless the user specifies otherwise.
8. Merge color avoidances from both the current request AND the profile's avoided colors.

## Few-Shot Examples

### Example 1
User: "smart casual dinner, London in October, budget around $200"
Profile: (new user, no history)
Output:
{{
    "occasion": "dinner",
    "formality_level": "Smart Casual",
    "social_context": "dinner out",
    "budget_min": 150.0,
    "budget_max": 200.0,
    "climate": "cool and possibly rainy, autumn in London",
    "location": "London",
    "season": "Fall",
    "gender_expression": "unspecified",
    "color_requests": [],
    "color_avoidances": [],
    "style_keywords": ["smart casual", "put-together", "dinner-appropriate"],
    "profile_conflicts": [],
    "profile_enhancements": [],
    "num_outfits_requested": 2,
    "special_instructions": null
}}

### Example 2
User: "first date, somewhere nice but not too formal, I want to look cool"
Profile:
  Style Identity: Minimalist streetwear with clean lines
  Preferred Colors: black, white, navy
  Avoided Colors: bright yellow, orange
  Avoided Fits: oversized, baggy
  Rejection Patterns: chunky sneakers (rejected 2x), graphic tees (rejected 1x)
Output:
{{
    "occasion": "first date at a nice restaurant",
    "formality_level": "Smart Casual",
    "social_context": "first date",
    "budget_min": null,
    "budget_max": null,
    "climate": null,
    "location": null,
    "season": null,
    "gender_expression": "unspecified",
    "color_requests": [],
    "color_avoidances": ["bright yellow", "orange"],
    "style_keywords": ["minimalist", "clean", "cool", "streetwear-influenced"],
    "profile_conflicts": [],
    "profile_enhancements": [
        "User prefers minimalist streetwear with clean lines - will lean into that",
        "Preferred palette of black/white/navy works well for a date look",
        "Avoiding oversized/baggy fits aligns with 'looking cool' in a polished way"
    ],
    "num_outfits_requested": 2,
    "special_instructions": "Avoid chunky sneakers and graphic tees per rejection history"
}}
"""


# ── Parser Agent ───────────────────────────────────────────────────────────

class IntentParser:
    """Parses natural language fashion requests into structured constraints."""

    def __init__(self, client: Optional[LLMClient] = None):
        # Light tier — structured extraction doesn't need the heavy model
        self.client = client or get_llm_client(tier="light")

    def parse(self, user_input: str, profile: Optional[TasteProfile] = None) -> ParsedIntent:
        """Parse a user's natural language request into structured constraints."""
        # Include profile whenever there is anything meaningful — gender counts even for session 0
        has_profile_data = profile and (
            profile.total_sessions > 0
            or profile.gender_expression not in ("unspecified", "", None)
            or bool(profile.color_preferences.preferred)
            or bool(profile.color_preferences.avoided)
        )
        if has_profile_data:
            profile_context = profile.get_profile_summary()
        else:
            profile_context = "(New user - no existing taste profile)"

        schema_str = json.dumps(ParsedIntent.model_json_schema(), indent=2)
        system_prompt = PARSER_SYSTEM_PROMPT.format(schema=schema_str)

        user_message = f"""## User Request
{user_input}

## Current Taste Profile
{profile_context}

Parse this request into the structured JSON format. Output ONLY valid JSON, no other text."""

        raw_output = self.client.complete(
            system=system_prompt,
            user=user_message,
            temperature=0.1,
            json_mode=True,
        )

        # Strip markdown fences if present
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            parsed_data = json.loads(cleaned)
            intent = ParsedIntent(**parsed_data)
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM output into ParsedIntent: {e}\n"
                f"Raw output: {raw_output}"
            )

        return intent

    def parse_to_dict(self, user_input: str,
                      profile: Optional[TasteProfile] = None) -> dict:
        """Parse and return as a plain dict."""
        return self.parse(user_input, profile).model_dump()
