"""LLM provider implementations.

Importing this package registers the LiteLLM provider, which is the
universal adapter every logical vendor (gemini, deepseek, ...)
resolves to.
"""

from paw.brain.providers.base import LLMProvider, ProviderRegistry
from paw.brain.providers.litellm import LiteLLMProvider

__all__ = ["LLMProvider", "ProviderRegistry", "LiteLLMProvider"]
