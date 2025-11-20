from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.config import Settings
from app.signal_logic import sanitize_signal_response


class LLMClient:
    """Thin wrapper around configured LLM provider."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model_name
        self.system_prompt = settings.llm_system_prompt
        self.confidence_threshold = settings.confidence_threshold

    def _openai_generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from openai import OpenAI  # lazy import to avoid import cost until used

        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _anthropic_generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from anthropic import Anthropic

        if not self.settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")

        client = Anthropic(api_key=self.settings.anthropic_api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=500,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        # Anthropics returns content as list of Text blocks
        content = "".join([block.text for block in message.content])
        return json.loads(content)

    def _build_system_prompt(self) -> str:
        if not self.system_prompt:
            raise RuntimeError("LLM_SYSTEM_PROMPT is not configured")
        return self.system_prompt

    def generate_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Calls the configured provider and returns sanitized, validated result."""
        if self.provider == "openai":
            raw = self._openai_generate(payload)
        elif self.provider == "anthropic":
            raw = self._anthropic_generate(payload)
        else:
            raise ValueError(f"Unsupported llm_provider: {self.provider}")

        raw["confidence_threshold"] = self.confidence_threshold
        return sanitize_signal_response(raw)
