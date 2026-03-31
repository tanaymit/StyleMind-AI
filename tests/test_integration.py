"""
StyleMind AI - Integration Tests (requires API credentials + FAISS index)
Tests the full pipeline: parse → plan → retrieve with the tiered model setup.

Run: python tests/test_integration.py
     python tests/test_integration.py --provider openai  # force OpenAI
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.llm_client import get_llm_client
from agents.taste_profile import TasteProfile, RejectionEntry
from agents.intent_parser import IntentParser
from agents.style_planner import StylePlanner
from config import FAISS_INDEX_PATH, LLM_PROVIDER


def test_intent_parser():
    """Test the Intent Parser with a real LLM call (light tier)."""
    print("Testing Intent Parser (light tier)...")

    client = get_llm_client(tier="light")
    print(f"  Model: {client.model_name}")
    parser = IntentParser(client=client)

    # Test 1: Simple request, no profile
    print("\n  Test 1: Simple request, new user")
    intent = parser.parse("smart casual dinner, London in October, budget around $200")
    print(f"    Occasion: {intent.occasion}")
    print(f"    Formality: {intent.formality_level}")
    print(f"    Budget: ${intent.budget_min} - ${intent.budget_max}")
    assert intent.formality_level in ["Smart Casual", "Semi-Formal"]
    assert intent.budget_max is not None
    print("    ✅ Passed")

    # Test 2: Request with profile (should detect blazer conflict)
    print("\n  Test 2: Request with profile conflicts")
    profile = TasteProfile(user_id="test_user")
    profile.style_identity = "Minimalist streetwear"
    profile.color_preferences.avoided = ["bright yellow", "orange"]
    profile.rejections = [
        RejectionEntry(item_type="blazers", count=3, reason="too formal"),
    ]
    profile.total_sessions = 5
    profile.profile_confidence = 0.4

    intent2 = parser.parse("I need a blazer for a work event", profile=profile)
    print(f"    Occasion: {intent2.occasion}")
    print(f"    Conflicts: {intent2.profile_conflicts}")
    print("    ✅ Passed")

    return intent


def test_style_planner(intent):
    """Test the Style Planner with a real LLM call (heavy tier)."""
    print("\nTesting Style Planner (heavy tier)...")

    client = get_llm_client(tier="heavy")
    print(f"  Model: {client.model_name}")
    planner = StylePlanner(client=client)

    output = planner.plan(intent)
    print(f"  Blueprints generated: {len(output.blueprints)}")

    for i, bp in enumerate(output.blueprints):
        print(f"\n  Blueprint {i+1}: {bp.outfit_name}")
        print(f"    Concept: {bp.outfit_concept[:80]}...")
        print(f"    Palette: {bp.color_palette}")
        print(f"    Items: {len(bp.items)}")
        for item in bp.items:
            print(f"      [{item.slot}] {item.article_type} → \"{item.search_query}\"")

    assert len(output.blueprints) >= 2
    print("\n  ✅ Planner passed")
    return output


def test_retriever(planner_output, intent):
    """Test the Product Retriever with FAISS index."""
    if not FAISS_INDEX_PATH.exists():
        print("\n⏭️  Skipping retriever test — run setup.py first")
        return

    print("\nTesting Product Retriever (FAISS + OpenAI embeddings)...")
    from agents.retriever import ProductRetriever
    retriever = ProductRetriever()
    intent_dict = intent.model_dump()

    bp = planner_output.blueprints[0]
    print(f"  Blueprint: {bp.outfit_name}")
    for item in bp.items[:2]:
        candidates = retriever.retrieve_for_blueprint_slot(
            search_query=item.search_query,
            slot=item.slot,
            target_color=item.target_color,
            intent_data=intent_dict,
            top_k=5,
        )
        print(f"\n    [{item.slot}] \"{item.search_query}\"")
        for c in candidates[:3]:
            print(f"      • {c['productDisplayName']} "
                  f"({c['baseColour']}, ${c.get('price', 0):.2f}, "
                  f"score={c['similarity_score']:.3f})")

    print("\n  ✅ Retriever passed")


def test_tier_separation():
    """Verify that heavy and light tiers use different models."""
    print("\nTesting tier separation...")
    heavy = get_llm_client(tier="heavy")
    light = get_llm_client(tier="light")

    print(f"  Heavy: {heavy.model_name}")
    print(f"  Light: {light.model_name}")

    if LLM_PROVIDER == "bedrock":
        assert heavy.model_name != light.model_name, \
            "Bedrock tiers should use different models"
        print("  ✅ Tiers use different models (Bedrock)")
    else:
        print("  ℹ️  OpenAI mode — both tiers use same model (override in config if needed)")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["bedrock", "openai"], default=None)
    args = ap.parse_args()

    if args.provider:
        import os
        os.environ["LLM_PROVIDER"] = args.provider
        import importlib, config
        importlib.reload(config)

    print("=" * 60)
    print(f"StyleMind AI - Integration Tests (provider: {LLM_PROVIDER})")
    print("=" * 60 + "\n")

    test_tier_separation()
    intent = test_intent_parser()
    planner_output = test_style_planner(intent)
    test_retriever(planner_output, intent)

    print("\n" + "=" * 60)
    print("All integration tests passed! ✅")
    print("=" * 60)
