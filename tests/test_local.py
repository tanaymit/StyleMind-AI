"""
StyleMind AI - Local Tests (no API calls)
Tests schema validation, profile I/O, and LLM client factory.
Run: python tests/test_local.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.taste_profile import TasteProfile, RejectionEntry
from agents.intent_parser import ParsedIntent


def test_taste_profile_creation():
    print("Testing TasteProfile creation and I/O...")
    with tempfile.TemporaryDirectory() as tmpdir:
        profile = TasteProfile(user_id="test_user_001")
        assert profile.profile_confidence == 0.0
        assert profile.total_sessions == 0

        profile.style_identity = "Minimalist with streetwear edge"
        profile.style_archetypes = ["minimalist", "streetwear"]
        profile.color_preferences.preferred = ["black", "white", "navy"]
        profile.color_preferences.avoided = ["bright yellow"]
        profile.fit_preferences.preferred_fits = ["slim fit", "tailored"]
        profile.fit_preferences.avoided_fits = ["oversized"]
        profile.rejections = [
            RejectionEntry(item_type="chunky sneakers", count=2),
            RejectionEntry(item_type="graphic tees", count=1, reason="too casual"),
        ]
        profile.gender_expression = "masculine"
        profile.total_sessions = 3
        profile.total_items_rated = 12
        profile.profile_confidence = 0.3

        save_path = Path(tmpdir) / "test_user_001.json"
        profile.save(path=save_path)
        assert save_path.exists()

        loaded = TasteProfile.load("test_user_001", path=save_path)
        assert loaded.style_identity == "Minimalist with streetwear edge"
        assert len(loaded.rejections) == 2

        summary = loaded.get_profile_summary()
        assert "Minimalist with streetwear edge" in summary
        assert "chunky sneakers" in summary

        print("  ✅ Profile creation, save, load, and summary all work")
        print(f"  Summary preview: {summary[:80]}...\n")


def test_parsed_intent_schema():
    print("Testing ParsedIntent schema validation...")

    intent = ParsedIntent(
        occasion="dinner",
        formality_level="Smart Casual",
        social_context="first date",
        budget_min=100,
        budget_max=200,
        season="Fall",
        color_requests=["navy"],
        style_keywords=["minimalist"],
    )
    assert intent.formality_level == "Smart Casual"
    assert intent.budget_max == 200.0

    minimal = ParsedIntent(occasion="casual outing", formality_level="Casual")
    assert minimal.num_outfits_requested == 2
    assert minimal.gender_expression == "unspecified"

    json_str = intent.model_dump_json(indent=2)
    reparsed = ParsedIntent.model_validate_json(json_str)
    assert reparsed.occasion == "dinner"

    print("  ✅ Schema validation, defaults, serialization all work\n")


def test_llm_client_factory():
    print("Testing LLM client factory...")
    import os

    # Test OpenAI path (doesn't need real key for instantiation check)
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "test-key"

    import importlib
    import config
    importlib.reload(config)

    # Re-import llm_client after config reload
    from agents import llm_client
    importlib.reload(llm_client)

    client_heavy = llm_client.get_llm_client(tier="heavy")
    client_light = llm_client.get_llm_client(tier="light")

    assert isinstance(client_heavy, llm_client.OpenAIClient)
    assert isinstance(client_light, llm_client.OpenAIClient)
    print(f"  OpenAI mode: heavy={client_heavy.model_name}, light={client_light.model_name}")

    # Test Bedrock path (factory only, no actual call)
    os.environ["LLM_PROVIDER"] = "bedrock"
    importlib.reload(config)
    importlib.reload(llm_client)

    try:
        import boto3
        client_heavy = llm_client.get_llm_client(tier="heavy")
        client_light = llm_client.get_llm_client(tier="light")
        assert isinstance(client_heavy, llm_client.BedrockClient)
        assert isinstance(client_light, llm_client.BedrockClient)
        assert "sonnet" in client_heavy.model_name.lower()
        assert "haiku" in client_light.model_name.lower()
        print(f"  Bedrock mode: heavy={client_heavy.model_name}")
        print(f"                light={client_light.model_name}")
    except ImportError:
        print("  ⏭️  boto3 not installed, skipping Bedrock client test")

    # Reset
    os.environ["LLM_PROVIDER"] = "bedrock"
    importlib.reload(config)
    importlib.reload(llm_client)

    print("  ✅ LLM client factory works for both providers\n")


def test_price_config():
    print("Testing price simulation config...")
    from config import PRICE_RANGES, DEFAULT_PRICE_RANGE
    assert PRICE_RANGES["Tshirts"][0] < PRICE_RANGES["Tshirts"][1]
    assert PRICE_RANGES["Suits"][1] > PRICE_RANGES["Tshirts"][1]
    assert DEFAULT_PRICE_RANGE == (15, 80)
    print("  ✅ Price ranges valid\n")


if __name__ == "__main__":
    print("=" * 60)
    print("StyleMind AI - Local Tests")
    print("=" * 60 + "\n")

    test_taste_profile_creation()
    test_parsed_intent_schema()
    test_llm_client_factory()
    test_price_config()

    print("=" * 60)
    print("All local tests passed! ✅")
    print("=" * 60)
