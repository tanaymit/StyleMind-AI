"""
StyleMind AI - Profile Updater Agent
Converts session feedback into structured profile diffs and applies them.
Uses the LIGHT tier model — structured extraction, not creative reasoning.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from agents.llm_client import get_llm_client, LLMClient
from agents.taste_profile import TasteProfile, RejectionEntry, SessionLogEntry


# ── Feedback Schemas ───────────────────────────────────────────────────────

class ItemFeedback(BaseModel):
    item_name: str
    item_type: str
    action: Literal["accepted", "rejected"]
    reason: Optional[str] = None


class SessionFeedback(BaseModel):
    outfit_name: Optional[str] = None
    outfit_accepted: bool = False
    item_feedback: list[ItemFeedback] = Field(default_factory=list)
    general_notes: Optional[str] = None


# ── Diff Schema ────────────────────────────────────────────────────────────

class ProfileDiff(BaseModel):
    style_identity_update: Optional[str] = None
    gender_expression: Optional[str] = Field(
        default=None,
        description="Set if gender is discernible: 'masculine', 'feminine', 'androgynous'. Null if unknown."
    )
    add_preferred_colors: list[str] = Field(default_factory=list)
    add_avoided_colors: list[str] = Field(default_factory=list)
    add_preferred_fits: list[str] = Field(default_factory=list)
    add_avoided_fits: list[str] = Field(default_factory=list)
    new_rejections: list[dict] = Field(
        default_factory=list,
        description='[{"item_type": "...", "reason": "..."}]'
    )
    confidence_delta: float = Field(default=0.05, ge=0.0, le=0.2)
    session_summary: str = ""


# ── System Prompt ──────────────────────────────────────────────────────────

UPDATER_SYSTEM = """You are the Profile Updater for StyleMind AI.

Given a user's session feedback, output a JSON diff to update their taste profile.

Output ONLY valid JSON with exactly these fields:
{
  "style_identity_update": null or "updated one-sentence style identity",
  "gender_expression": null or "masculine" or "feminine" or "androgynous",
  "add_preferred_colors": [],
  "add_avoided_colors": [],
  "add_preferred_fits": [],
  "add_avoided_fits": [],
  "new_rejections": [{"item_type": "...", "reason": "..."}],
  "confidence_delta": 0.05,
  "session_summary": "one sentence"
}

Rules:
- Only add NEW information not already in the profile
- gender_expression: infer from the request context (e.g. "men's", "women's", outfit type). Null if ambiguous.
- new_rejections: only for items the user explicitly rejected
- confidence_delta: 0.08 if outfit accepted + items rated, 0.05 normal, 0.02 if no feedback
- style_identity_update: only if the session reveals something meaningfully new about style
- session_summary: one sentence describing the request and outcome
- Output ONLY the JSON object, no markdown, no extra text"""


# ── Updater Agent ──────────────────────────────────────────────────────────

class ProfileUpdater:
    """Converts session feedback into a ProfileDiff and applies it to the profile."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_llm_client(tier="light")

    def compute_diff(
        self,
        profile: TasteProfile,
        feedback: SessionFeedback,
        request_summary: str,
    ) -> ProfileDiff:
        current_summary = profile.get_profile_summary()

        accepted = [f.item_name for f in feedback.item_feedback if f.action == "accepted"]
        rejected = [
            f"{f.item_name} ({f.reason or 'no reason given'})"
            for f in feedback.item_feedback if f.action == "rejected"
        ]

        user_msg = f"""Current Profile:
{current_summary}

Session:
- Request: {request_summary}
- Outfit accepted: {feedback.outfit_accepted} (outfit: {feedback.outfit_name or 'N/A'})
- Items accepted: {accepted or 'none'}
- Items rejected: {rejected or 'none'}
- Notes: {feedback.general_notes or 'none'}

Generate the profile diff JSON:"""

        raw = self.client.complete(
            system=UPDATER_SYSTEM,
            user=user_msg,
            temperature=0.1,
            json_mode=True,
            max_tokens=512,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

        diff_data = json.loads(cleaned.strip())
        return ProfileDiff(**diff_data)

    def apply_diff(
        self,
        profile: TasteProfile,
        diff: ProfileDiff,
        feedback: SessionFeedback,
        request_summary: str,
    ) -> TasteProfile:
        if diff.style_identity_update:
            profile.style_identity = diff.style_identity_update

        if diff.gender_expression and profile.gender_expression == "unspecified":
            profile.gender_expression = diff.gender_expression

        for c in diff.add_preferred_colors:
            if c not in profile.color_preferences.preferred:
                profile.color_preferences.preferred.append(c)

        for c in diff.add_avoided_colors:
            if c not in profile.color_preferences.avoided:
                profile.color_preferences.avoided.append(c)

        for f in diff.add_preferred_fits:
            if f not in profile.fit_preferences.preferred_fits:
                profile.fit_preferences.preferred_fits.append(f)

        for f in diff.add_avoided_fits:
            if f not in profile.fit_preferences.avoided_fits:
                profile.fit_preferences.avoided_fits.append(f)

        for rej in diff.new_rejections:
            item_type = rej.get("item_type", "")
            if not item_type:
                continue
            existing = next(
                (r for r in profile.rejections if r.item_type == item_type), None
            )
            if existing:
                existing.count += 1
                existing.last_rejected = datetime.now().isoformat()
            else:
                profile.rejections.append(RejectionEntry(
                    item_type=item_type,
                    reason=rej.get("reason"),
                ))

        profile.profile_confidence = min(
            1.0, profile.profile_confidence + diff.confidence_delta
        )
        profile.total_sessions += 1
        profile.total_items_rated += len(feedback.item_feedback)

        profile.session_log.append(SessionLogEntry(
            session_id=f"session_{profile.total_sessions}",
            request_summary=request_summary,
            items_accepted=[f.item_name for f in feedback.item_feedback if f.action == "accepted"],
            items_rejected=[f.item_name for f in feedback.item_feedback if f.action == "rejected"],
            feedback_notes=diff.session_summary,
        ))

        profile.save()
        return profile

    def update(
        self,
        profile: TasteProfile,
        feedback: SessionFeedback,
        request_summary: str,
    ) -> TasteProfile:
        """Full update: compute diff → apply → save. Returns the updated profile."""
        diff = self.compute_diff(profile, feedback, request_summary)
        return self.apply_diff(profile, diff, feedback, request_summary)
