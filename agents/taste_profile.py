"""
StyleMind AI - Taste Profile Schema
Persistent user taste profile that evolves across interactions.

The profile is stored as a JSON file per user and read at the start
of every session by the Intent Parser and Style Planning Agent.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from config import PROFILES_DIR


# ── Sub-schemas ────────────────────────────────────────────────────────────

class ColorPreferences(BaseModel):
    """User's color preferences and rules."""
    preferred: list[str] = Field(default_factory=list,
        description="Colors the user gravitates toward")
    avoided: list[str] = Field(default_factory=list,
        description="Colors the user has explicitly rejected")
    preferred_palettes: list[str] = Field(default_factory=list,
        description="General palette descriptions, e.g. 'earth tones', 'monochrome'")


class FitPreferences(BaseModel):
    """Fit and silhouette preferences."""
    preferred_fits: list[str] = Field(default_factory=list,
        description="E.g., 'slim fit', 'oversized', 'tailored'")
    avoided_fits: list[str] = Field(default_factory=list,
        description="E.g., 'skinny', 'baggy'")
    texture_preferences: list[str] = Field(default_factory=list,
        description="E.g., 'cotton', 'linen', 'avoids polyester'")


class RejectionEntry(BaseModel):
    """A tracked rejection pattern."""
    item_type: str = Field(description="E.g., 'chunky sneakers', 'blazers'")
    count: int = Field(default=1, description="Number of times rejected")
    last_rejected: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of most recent rejection"
    )
    reason: Optional[str] = Field(default=None,
        description="User's stated reason, if provided")


class SessionLogEntry(BaseModel):
    """Summary of a single interaction session."""
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_summary: str = Field(description="What the user asked for")
    items_accepted: list[str] = Field(default_factory=list)
    items_rejected: list[str] = Field(default_factory=list)
    feedback_notes: Optional[str] = Field(default=None)


# ── Main Profile ───────────────────────────────────────────────────────────

class TasteProfile(BaseModel):
    """
    Complete user taste profile.

    This is the central data structure that makes StyleMind AI stateful.
    It is read by the Intent Parser before generating constraints, and
    by the Style Planning Agent before creating outfit blueprints.

    The Profile Updater agent (end-of-semester work) will write structured
    diffs to this profile after each session.
    """

    # Identity
    user_id: str = Field(description="Unique user identifier")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat())

    # Style identity
    style_identity: str = Field(
        default="Not yet established",
        description=(
            "Free-text summary of the user's style identity, e.g., "
            "'Minimalist Scandinavian with occasional streetwear edge'"
        )
    )
    style_archetypes: list[str] = Field(
        default_factory=list,
        description="E.g., ['minimalist', 'smart casual', 'streetwear']"
    )

    # Preferences
    color_preferences: ColorPreferences = Field(
        default_factory=ColorPreferences)
    fit_preferences: FitPreferences = Field(
        default_factory=FitPreferences)

    # Demographics / context
    gender_expression: str = Field(
        default="unspecified",
        description="How the user wants to be styled: 'masculine', 'feminine', 'androgynous', 'unspecified'"
    )
    typical_budget_range: Optional[str] = Field(
        default=None,
        description="E.g., '$50-150 per outfit'"
    )
    climate_context: Optional[str] = Field(
        default=None,
        description="E.g., 'lives in London, mostly temperate'"
    )

    # Rejection patterns
    rejections: list[RejectionEntry] = Field(
        default_factory=list,
        description="Tracked rejection patterns with counts"
    )

    # Session history (lightweight)
    session_log: list[SessionLogEntry] = Field(
        default_factory=list,
        description="Summary log of past sessions"
    )

    # Meta
    profile_confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description=(
            "Estimated calibration of the profile (0.0 = brand new, "
            "1.0 = highly confident). Increases with interactions."
        )
    )
    total_sessions: int = Field(default=0)
    total_items_rated: int = Field(default=0)

    # ── I/O Methods ────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Save profile to disk as JSON."""
        if path is None:
            path = PROFILES_DIR / f"{self.user_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, user_id: str, path: Optional[Path] = None) -> TasteProfile:
        """Load profile from disk, or create a new one if it doesn't exist."""
        if path is None:
            path = PROFILES_DIR / f"{user_id}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        else:
            return cls(user_id=user_id)

    def get_profile_summary(self) -> str:
        """
        Generate a concise natural language summary for injection into
        LLM prompts. This is what the Intent Parser and Style Planner see.
        """
        parts = [f"Style Identity: {self.style_identity}"]

        if self.style_archetypes:
            parts.append(f"Style Archetypes: {', '.join(self.style_archetypes)}")

        if self.color_preferences.preferred:
            parts.append(f"Preferred Colors: {', '.join(self.color_preferences.preferred)}")
        if self.color_preferences.avoided:
            parts.append(f"Avoided Colors: {', '.join(self.color_preferences.avoided)}")

        if self.fit_preferences.preferred_fits:
            parts.append(f"Preferred Fits: {', '.join(self.fit_preferences.preferred_fits)}")
        if self.fit_preferences.avoided_fits:
            parts.append(f"Avoided Fits: {', '.join(self.fit_preferences.avoided_fits)}")

        if self.rejections:
            top_rejections = sorted(self.rejections, key=lambda r: r.count, reverse=True)[:5]
            rej_strs = [f"{r.item_type} (rejected {r.count}x)" for r in top_rejections]
            parts.append(f"Rejection Patterns: {', '.join(rej_strs)}")

        if self.gender_expression != "unspecified":
            parts.append(f"Gender Expression: {self.gender_expression}")
        if self.typical_budget_range:
            parts.append(f"Typical Budget: {self.typical_budget_range}")
        if self.climate_context:
            parts.append(f"Climate: {self.climate_context}")

        parts.append(f"Profile Confidence: {self.profile_confidence:.0%} "
                      f"({self.total_sessions} sessions, "
                      f"{self.total_items_rated} items rated)")

        return "\n".join(parts)


# ── Factory helper ─────────────────────────────────────────────────────────

def create_new_profile(user_id: str, **kwargs) -> TasteProfile:
    """Create and save a fresh profile."""
    profile = TasteProfile(user_id=user_id, **kwargs)
    profile.save()
    return profile
