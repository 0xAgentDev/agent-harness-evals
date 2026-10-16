"""Settings: everything the harness needs comes from the environment.

The harness talks to any OpenAI-compatible /chat/completions endpoint, so it
works with OpenAI, a local Ollama server, or a self-hosted gateway like
OmniRoute without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("AGENT_MODEL", "gpt-4o-mini"))
    max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_STEPS", "20")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("AGENT_TEMPERATURE", "0.2")))
    # USD per 1K tokens, used only for eval cost estimates. Override per model.
    price_in_per_1k: float = field(default_factory=lambda: float(os.environ.get("AGENT_PRICE_IN_PER_1K", "0.00015")))
    price_out_per_1k: float = field(default_factory=lambda: float(os.environ.get("AGENT_PRICE_OUT_PER_1K", "0.0006")))
    shell_timeout_s: int = field(default_factory=lambda: int(os.environ.get("AGENT_SHELL_TIMEOUT_S", "30")))

    def require_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Point OPENAI_BASE_URL at any "
                "OpenAI-compatible endpoint and set a key for it."
            )
