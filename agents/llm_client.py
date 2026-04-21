"""
StyleMind AI - LLM Client
Unified interface for calling LLMs across providers (Bedrock / OpenAI).

Supports two tiers:
  - "heavy": Sonnet 4.6 (creative reasoning, planning, evaluation)
  - "light": Haiku 4.5 (structured extraction, parsing, updates)

Usage:
    from agents.llm_client import get_llm_client

    client = get_llm_client(tier="heavy")
    response = client.complete(
        system="You are a helpful assistant.",
        user="Plan an outfit for a dinner date.",
        temperature=0.7,
        json_mode=True,
    )
    print(response)  # raw string (JSON or text)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional

from config import (
    AWS_REGION,
    BEDROCK_MODEL_HEAVY,
    BEDROCK_MODEL_LIGHT,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_LLM_MODEL,
)


class LLMClient(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Send a prompt and return the raw text response."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for logging."""
        ...


# ── Bedrock (Claude via AWS) ──────────────────────────────────────────────

class BedrockClient(LLMClient):
    """
    Calls Claude models on AWS Bedrock using the Converse API.
    Uses boto3 — credentials come from env vars, ~/.aws/credentials, or IAM role.
    """

    def __init__(self, model_id: str, region: str = AWS_REGION):
        import boto3
        from botocore.config import Config
        self._model_id = model_id
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                read_timeout=120,
                connect_timeout=15,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )

    @property
    def model_name(self) -> str:
        return self._model_id

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        # Build request using Bedrock Converse API (provider-agnostic)
        messages = [{"role": "user", "content": [{"text": user}]}]
        system_prompts = [{"text": system}]

        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
        }

        kwargs = {
            "modelId": self._model_id,
            "messages": messages,
            "system": system_prompts,
            "inferenceConfig": inference_config,
        }

        response = self._client.converse(**kwargs)

        # Extract text from response
        output = response["output"]["message"]["content"]
        text = "".join(
            block["text"] for block in output if "text" in block
        )

        return text.strip()


# ── OpenAI ─────────────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """Calls OpenAI models via the openai SDK."""

    def __init__(self, model: str = OPENAI_LLM_MODEL, api_key: str = OPENAI_API_KEY,
             base_url: str = None):
        from openai import OpenAI
        from config import OPENAI_BASE_URL
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url or OPENAI_BASE_URL)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()


# ── Factory ────────────────────────────────────────────────────────────────

def get_llm_client(tier: str = "heavy") -> LLMClient:
    """
    Get an LLM client for the specified tier.

    Args:
        tier: "heavy" (Sonnet 4.6 — planning, generation, evaluation)
              "light" (Haiku 4.5 — parsing, profile updates)

    Returns:
        LLMClient instance configured for the active provider.
    """
    if LLM_PROVIDER == "bedrock":
        model_id = BEDROCK_MODEL_HEAVY if tier == "heavy" else BEDROCK_MODEL_LIGHT
        return BedrockClient(model_id=model_id)

    elif LLM_PROVIDER == "openai":
        # OpenAI doesn't have a built-in tier split, so we use the same model.
        # You could map tiers to different models (e.g., gpt-5 vs gpt-4o-mini).
        return OpenAIClient(model=OPENAI_LLM_MODEL)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
            "Set LLM_PROVIDER to 'bedrock' or 'openai' in .env"
        )
