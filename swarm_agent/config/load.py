"""Typed configuration loading and file-access scope/permission models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.json"


@dataclass(frozen=True)
class FileScope:
    """A named set of filesystem permissions."""

    roots: list[str] = field(default_factory=list)
    read: bool = True
    list: bool = True


@dataclass(frozen=True)
class FileAccessConfig:
    """Runtime-controlled file access policy.

    The model selects a scope by name when calling ``read_file`` or
    ``list_dir``. The roots and permissions for each scope are owned
    by config; the model cannot expand them.
    """

    default_scope: str = "kb"
    scopes: dict[str, FileScope] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "FileAccessConfig":
        scopes: dict[str, FileScope] = {}
        for name, raw_scope in (data.get("scopes") or {}).items():
            if not isinstance(raw_scope, dict):
                continue
            scopes[name] = FileScope(
                roots=list(raw_scope.get("roots") or []),
                read=bool(raw_scope.get("read", True)),
                list=bool(raw_scope.get("list", True)),
            )
        return cls(
            default_scope=str(data.get("default_scope") or "kb"),
            scopes=scopes,
        )


@dataclass(frozen=True)
class InterfaceConfig:
    """Config for a runtime interface."""

    enabled: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    """Config for the scheduler service."""

    enabled: bool = True
    poll_seconds: int = 10


@dataclass(frozen=True)
class AgentSpec:
    """Runtime identity for one Swarm agent."""

    id: str
    description: str = ""
    shared_soul: str | None = None
    agent_profile: str | None = None
    model_strategy: list[str] = field(default_factory=list)
    can_delegate: list[str] = field(default_factory=list)
    max_tool_rounds: int = 8

    @classmethod
    def from_dict(
        cls,
        agent_id: str,
        data: dict,
        *,
        defaults: dict | None = None,
    ) -> "AgentSpec":
        defaults = defaults or {}
        return cls(
            id=agent_id,
            description=str(data.get("description") or ""),
            shared_soul=data.get("shared_soul", defaults.get("shared_soul")),
            agent_profile=data.get("agent_profile", defaults.get("agent_profile")),
            model_strategy=list(
                data.get("model_strategy")
                or defaults.get("model_strategy")
                or []
            ),
            can_delegate=list(data.get("can_delegate") or []),
            max_tool_rounds=int(
                data.get("max_tool_rounds")
                or defaults.get("max_tool_rounds")
                or 8
            ),
        )


@dataclass(frozen=True)
class ExternalAgentSpec:
    """CLI-backed external agent exposed through a Swarm tool."""

    id: str
    command: list[str] = field(default_factory=list)
    timeout_seconds: int = 900
    max_output_chars: int = 20000
    prompt_mode: str = "stdin"
    description: str = ""

    @classmethod
    def from_dict(cls, agent_id: str, data: dict) -> "ExternalAgentSpec":
        return cls(
            id=agent_id,
            command=list(data.get("command") or []),
            timeout_seconds=int(data.get("timeout_seconds", 900)),
            max_output_chars=int(data.get("max_output_chars", 20000)),
            prompt_mode=str(data.get("prompt_mode") or "stdin"),
            description=str(data.get("description") or ""),
        )


@dataclass(frozen=True)
class SwarmConfig:
    """Top-level runtime config."""

    interfaces: dict[str, InterfaceConfig] = field(default_factory=dict)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    file_access: FileAccessConfig = field(default_factory=FileAccessConfig)
    models: dict = field(default_factory=dict)
    model_strategy: list[str] = field(default_factory=list)
    shared_soul: str | None = None
    agent_profile: str | None = None
    retry: dict = field(default_factory=dict)
    default_agent: str = "default"
    agents: dict[str, AgentSpec] = field(default_factory=dict)
    external_agents: dict[str, ExternalAgentSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SwarmConfig":
        interfaces = {
            name: InterfaceConfig(enabled=bool(settings.get("enabled", False)))
            for name, settings in data.get("interfaces", {}).items()
            if isinstance(settings, dict)
        }
        scheduler_data = data.get("scheduler", {})
        scheduler = SchedulerConfig(
            enabled=bool(scheduler_data.get("enabled", True)),
            poll_seconds=int(scheduler_data.get("poll_seconds", 10)),
        )
        default_agent = str(data.get("default_agent") or "default")
        defaults = {
            "shared_soul": data.get("shared_soul"),
            "agent_profile": data.get("agent_profile"),
            "model_strategy": list(data.get("model_strategy") or []),
            "max_tool_rounds": int(data.get("max_tool_rounds") or 8),
        }

        agents_data = data.get("agents") or {}
        agents: dict[str, AgentSpec] = {}
        if isinstance(agents_data, dict) and agents_data:
            for agent_id, raw_agent in agents_data.items():
                if isinstance(raw_agent, dict):
                    agents[agent_id] = AgentSpec.from_dict(
                        agent_id,
                        raw_agent,
                        defaults=defaults,
                    )
        else:
            agents[default_agent] = AgentSpec.from_dict(
                default_agent,
                {
                    "description": "Default Swarm agent synthesized from legacy config.",
                },
                defaults=defaults,
            )

        external_agents_data = data.get("external_agents") or {}
        external_agents = {
            agent_id: ExternalAgentSpec.from_dict(agent_id, raw_agent)
            for agent_id, raw_agent in external_agents_data.items()
            if isinstance(raw_agent, dict)
        } if isinstance(external_agents_data, dict) else {}

        return cls(
            interfaces=interfaces,
            scheduler=scheduler,
            file_access=FileAccessConfig.from_dict(data.get("file_access", {})),
            models=dict(data.get("models") or {}),
            model_strategy=list(data.get("model_strategy") or []),
            shared_soul=data.get("shared_soul"),
            agent_profile=data.get("agent_profile"),
            retry=dict(data.get("retry") or {}),
            default_agent=default_agent,
            agents=agents,
            external_agents=external_agents,
        )

    def enabled_interfaces(self) -> list[str]:
        """Return the names of interfaces enabled in config."""
        return [name for name, config in self.interfaces.items() if config.enabled]

    def get_agent(self, agent_id: str | None = None) -> AgentSpec:
        """Return a configured agent, falling back to the default agent."""
        resolved_id = agent_id or self.default_agent
        if resolved_id in self.agents:
            return self.agents[resolved_id]
        if self.default_agent in self.agents:
            return self.agents[self.default_agent]
        raise ValueError(f"Agent '{resolved_id}' is not configured.")


def load_raw_config(path: Path = CONFIG_PATH) -> dict:
    """Load raw JSON config used by the runtime and Brain.

    Environment overrides:
        SWARM_SHARED_SOUL: shared soul prompt resource path.
        SWARM_AGENT_PROFILE: agent profile / KB entrypoint path.
        SWARM_KNOWLEDGE_ROOTS: os.pathsep-separated roots for the ``kb`` scope.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Legacy ``knowledge.allowed_roots`` is mapped into the ``kb`` scope so
    # older configs keep working. New configs should use ``file_access``.
    if "file_access" not in data and data.get("knowledge", {}).get("allowed_roots"):
        data["file_access"] = {
            "default_scope": "kb",
            "scopes": {
                "kb": {
                    "roots": data["knowledge"]["allowed_roots"],
                    "read": True,
                    "list": True,
                }
            },
        }

    if shared_soul := os.getenv("SWARM_SHARED_SOUL"):
        data["shared_soul"] = shared_soul
        default_agent = data.get("default_agent")
        if default_agent and isinstance(data.get("agents"), dict):
            data["agents"].setdefault(default_agent, {})["shared_soul"] = shared_soul

    if profile := os.getenv("SWARM_AGENT_PROFILE"):
        data["agent_profile"] = profile
        default_agent = data.get("default_agent")
        if default_agent and isinstance(data.get("agents"), dict):
            data["agents"].setdefault(default_agent, {})["agent_profile"] = profile

    if roots := os.getenv("SWARM_KNOWLEDGE_ROOTS"):
        data.setdefault("file_access", {}).setdefault("scopes", {}).setdefault(
            "kb", {}
        )["roots"] = [r for r in roots.split(os.pathsep) if r]
        data["file_access"].setdefault("default_scope", "kb")

    return data


_config_cache: dict[str, SwarmConfig] = {}


def load_swarm_config(path: Path = CONFIG_PATH) -> SwarmConfig:
    """Load typed runtime config (cached per path for the process lifetime)."""
    key = str(path)
    if key not in _config_cache:
        _config_cache[key] = SwarmConfig.from_dict(load_raw_config(path))
    return _config_cache[key]
