"""Abstract LLM provider and the small provider registry."""

import abc
from typing import Dict, List, Optional, Tuple, Type

from swarm_agent.contracts.types import (
    BrainDecision,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class LLMProvider(abc.ABC):
    """Base class for all LLM providers.

    Today there is a single concrete provider (``LiteLLMProvider``) that
    routes every model through LiteLLM. The abstraction is kept so the
    Brain doesn't have to know about LiteLLM directly, and so future
    custom providers (e.g. a local model) can plug in.
    """

    def supports_native_tools(self) -> bool:
        return False

    @abc.abstractmethod
    async def generate_with_tools(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        tool_definitions: List[ToolDefinition],
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> Tuple[BrainDecision, dict]:
        """Make an API call with native tool definitions."""

    @abc.abstractmethod
    async def continue_with_tool_results(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        tool_calls: List[ToolCall],
        tool_results: List[ToolResult],
        tool_definitions: List[ToolDefinition],
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> Tuple[BrainDecision, dict]:
        """Send tool results back to continue the conversation."""

    @abc.abstractmethod
    async def generate_decision(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        response_schema: dict,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """JSON-mode fallback for models without native tool calling."""

    @abc.abstractmethod
    def is_retryable(self, exc: Exception) -> bool:
        """Whether the brain should retry with a different model on this error."""


class ProviderRegistry:
    """Registry of provider classes by short name.

    With LiteLLM in place there's normally a single registered provider
    (``litellm``). Logical provider names like ``gemini``, ``deepseek``,
    ``siliconflow`` still appear in ``config.json`` so the brain can
    skip a failed vendor when retrying. They all resolve to the LiteLLM
    provider.
    """

    _registry: Dict[str, Type[LLMProvider]] = {}
    _instances: Dict[str, LLMProvider] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(provider_class: Type[LLMProvider]):
            cls._registry[name] = provider_class
            return provider_class
        return wrapper

    @classmethod
    def get_provider(cls, name: str) -> LLMProvider:
        # Logical vendor names (gemini, deepseek, siliconflow, ...) all
        # share the same LiteLLM-backed provider instance.
        if name not in cls._registry:
            name = "litellm"
        if name not in cls._instances:
            cls._instances[name] = cls._registry[name]()
        return cls._instances[name]
