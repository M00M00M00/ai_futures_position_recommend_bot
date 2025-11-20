from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.config import Settings
from app.signal_logic import sanitize_signal_response


class LLMClient:
    """Thin wrapper around configured LLM provider."""

    def __init__(self, settings: Settings, override_model: Optional[str] = None):
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.model = override_model or settings.llm_model_name
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
        if self.system_prompt and self.system_prompt.strip():
            return self.system_prompt
        if self.settings.llm_system_prompt_file:
            try:
                with open(self.settings.llm_system_prompt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    return content
            except FileNotFoundError:
                raise RuntimeError(f"LLM system prompt file not found: {self.settings.llm_system_prompt_file}")
        raise RuntimeError("LLM system prompt is not configured")

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

    def set_model(self, model_name: str) -> None:
        self.model = model_name
