#!/usr/bin/env python3
"""
StyleMind AI - Midpoint Demo
Demonstrates the full pipeline: Input → Parse → Plan → Retrieve

Usage:
    python demo.py                  # Full pipeline (needs FAISS index)
    python demo.py --no-retrieval   # Parser + Planner only
    python demo.py --provider openai  # Override provider
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.llm_client import get_llm_client
from agents.taste_profile import TasteProfile, RejectionEntry
from agents.intent_parser import IntentParser
from agents.style_planner import StylePlanner


def create_demo_profile() -> TasteProfile:
    """Create a sample user profile for demo purposes."""
    profile = TasteProfile(user_id="demo_user")
    profile.style_identity = "Clean minimalist with a slight streetwear influence"
    profile.style_archetypes = ["minimalist", "smart casual", "streetwear"]
    profile.color_preferences.preferred = ["black", "white", "navy", "charcoal"]
    profile.color_preferences.avoided = ["bright yellow", "neon green"]
    profile.color_preferences.preferred_palettes = ["monochrome", "earth tones"]
    profile.fit_preferences.preferred_fits = ["slim fit", "tailored", "clean lines"]
    profile.fit_preferences.avoided_fits = ["oversized", "baggy"]
    profile.fit_preferences.texture_preferences = ["cotton", "linen", "leather accents"]
    profile.gender_expression = "masculine"
    profile.typical_budget_range = "$100-250 per outfit"
    profile.climate_context = "Temperate, 4 seasons (Pittsburgh, PA)"
    profile.rejections = [
        RejectionEntry(item_type="chunky sneakers", count=3, reason="too bulky"),
        RejectionEntry(item_type="graphic tees", count=2, reason="too casual for my vibe"),
        RejectionEntry(item_type="cargo pants", count=1, reason="not clean enough"),
    ]
    profile.total_sessions = 7
    profile.total_items_rated = 28
    profile.profile_confidence = 0.45
    return profile


def run_demo(use_retrieval: bool = True):
    test_queries = [
        "Smart casual dinner date, somewhere nice in Pittsburgh, October evening, budget around $180",
        "Job interview at a tech startup - want to look sharp but not overdressed",
        "Weekend brunch with friends, relaxed but put-together, sunny day",
    ]

    profile = create_demo_profile()

    print("=" * 70)
    print("STYLEMIND AI — MIDPOINT DEMO")
    print("=" * 70)

    # Show which models are being used
    light_client = get_llm_client(tier="light")
    heavy_client = get_llm_client(tier="heavy")
    print(f"\n🔧 Provider config:")
    print(f"   Light tier (parser):  {light_client.model_name}")
    print(f"   Heavy tier (planner): {heavy_client.model_name}")

    print(f"\n📋 User Profile Summary:")
    print("-" * 40)
    print(profile.get_profile_summary())
    print()

    parser = IntentParser(client=light_client)
    planner = StylePlanner(client=heavy_client)

    retriever = None
    if use_retrieval:
        try:
            from agents.retriever import ProductRetriever
            from config import FAISS_INDEX_PATH
            if FAISS_INDEX_PATH.exists():
                retriever = ProductRetriever()
                print("✅ Retriever loaded (FAISS index found)\n")
            else:
                print("⚠️  FAISS index not found — skipping retrieval step\n")
        except Exception as e:
            print(f"⚠️  Retriever not available: {e}\n")

    for i, query in enumerate(test_queries):
        print("=" * 70)
        print(f"QUERY {i+1}: \"{query}\"")
        print("=" * 70)

        # Step 1: Parse
        print("\n🔍 Step 1: Intent Parsing (light tier)...")
        try:
            intent = parser.parse(query, profile=profile)
            print(f"  Occasion:     {intent.occasion}")
            print(f"  Formality:    {intent.formality_level}")
            print(f"  Budget:       ${intent.budget_min or '?'} - ${intent.budget_max or '?'}")
            print(f"  Season:       {intent.season or 'not specified'}")
            print(f"  Location:     {intent.location or 'not specified'}")
            print(f"  Style keys:   {intent.style_keywords}")
            if intent.profile_conflicts:
                print(f"  ⚠️ Conflicts:  {intent.profile_conflicts}")
            if intent.profile_enhancements:
                print(f"  ✨ Enhanced:   {intent.profile_enhancements[:2]}")
        except Exception as e:
            print(f"  ❌ Parser error: {e}")
            continue

        # Step 2: Plan
        print("\n🎨 Step 2: Style Planning (heavy tier, chain-of-thought)...")
        try:
            plan = planner.plan(intent, profile=profile)
            print(f"  Analysis: {plan.occasion_analysis[:120]}...")
            print(f"  Profile integration: {plan.profile_integration[:120]}...")

            for j, bp in enumerate(plan.blueprints):
                print(f"\n  Outfit {j+1}: \"{bp.outfit_name}\"")
                print(f"    Concept: {bp.outfit_concept}")
                print(f"    Palette: {bp.color_palette}")
                print(f"    Signal:  {bp.social_signal}")
                for item in bp.items:
                    print(f"    • [{item.slot:10s}] {item.article_type} ({item.target_color})")
                    print(f"      Reasoning: {item.reasoning[:90]}...")
                    print(f"      Search:    \"{item.search_query}\"")
        except Exception as e:
            print(f"  ❌ Planner error: {e}")
            continue

        # Step 3: Retrieve
        if retriever and plan:
            print(f"\n🔎 Step 3: Product Retrieval (OpenAI embeddings + FAISS)...")
            intent_dict = intent.model_dump()
            for j, bp in enumerate(plan.blueprints[:1]):
                print(f"\n  Retrieving for \"{bp.outfit_name}\":")
                for item in bp.items:
                    try:
                        candidates = retriever.retrieve_for_blueprint_slot(
                            search_query=item.search_query,
                            slot=item.slot,
                            target_color=item.target_color,
                            intent_data=intent_dict,
                            top_k=5,
                        )
                        print(f"\n    [{item.slot}] Top 3 matches:")
                        for c in candidates[:3]:
                            print(f"      • {c['productDisplayName']}")
                            print(f"        {c['baseColour']}, ${c.get('price', 'N/A'):.2f} "
                                  f"(score: {c['similarity_score']:.3f})")
                    except Exception as e:
                        print(f"    [{item.slot}] Retrieval error: {e}")

        print("\n")

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--provider", choices=["bedrock", "openai"], default=None,
                    help="Override LLM_PROVIDER from .env")
    args = ap.parse_args()

    if args.provider:
        import os
        os.environ["LLM_PROVIDER"] = args.provider
        # Re-import config to pick up the override
        import importlib
        import config
        importlib.reload(config)

    run_demo(use_retrieval=not args.no_retrieval)
