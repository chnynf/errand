"""Tests for config loading and environment overrides."""

import json
from pathlib import Path

from paw.config import load_raw_config, load_paw_config


def test_load_raw_config_applies_prompt_resource_overrides(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"shared_soul": "soul-file", "agent_profile": "from-file"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PAW_SHARED_SOUL", "soul-env")
    monkeypatch.setenv("PAW_AGENT_PROFILE", "from-env")

    data = load_raw_config(config_path)
    assert data["shared_soul"] == "soul-env"
    assert data["agent_profile"] == "from-env"


def test_load_raw_config_applies_knowledge_roots_override(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "file_access": {
                    "default_scope": "kb",
                    "scopes": {"kb": {"roots": ["from-file"]}},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PAW_KNOWLEDGE_ROOTS", "root-a:root-b")

    data = load_raw_config(config_path)
    assert data["file_access"]["scopes"]["kb"]["roots"] == ["root-a", "root-b"]


def test_load_paw_config_synthesizes_legacy_default_agent(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "shared_soul": "soul",
                "agent_profile": "profile",
                "model_strategy": ["m1"],
            }
        ),
        encoding="utf-8",
    )

    config = load_paw_config(config_path)
    agent = config.get_agent()
    assert config.default_agent == "default"
    assert agent.shared_soul == "soul"
    assert agent.agent_profile == "profile"
    assert agent.model_strategy == ["m1"]


def test_load_paw_config_parses_session_config(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"model_strategy": ["m1"], "sessions": {"idle_boundary_hours": 12}}),
        encoding="utf-8",
    )

    config = load_paw_config(config_path)

    assert config.sessions.idle_boundary_hours == 12.0


def test_load_paw_config_parses_agents_and_external_agents(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "default_agent": "generalist",
                "model_strategy": ["fallback"],
                "agents": {
                    "generalist": {
                        "agent_profile": "generalist-profile",
                        "model_strategy": ["fast"],
                        "can_delegate": ["cursor"],
                    },
                    "applied-scientist": {
                        "agent_profile": "as-profile",
                    },
                },
                "external_agents": {
                    "cursor": {
                        "command": ["cursor-agent", "--print"],
                        "prompt_mode": "stdin",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_paw_config(config_path)
    assert config.default_agent == "generalist"
    assert config.get_agent().agent_profile == "generalist-profile"
    assert config.get_agent("applied-scientist").model_strategy == ["fallback"]
    assert config.get_agent("generalist").can_delegate == ["cursor"]
    assert config.external_agents["cursor"].command == ["cursor-agent", "--print"]
    assert config.external_agents["cursor"].prompt_mode == "stdin"


def test_load_paw_config_parses_per_agent_reasoning_effort(tmp_path: Path):
    """Effort is read only from each agent; unset means provider default (None)."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "default_agent": "generalist",
                "model_strategy": ["fast"],
                "agents": {
                    "generalist": {"agent_profile": "g"},
                    "applied-scientist": {
                        "agent_profile": "as",
                        "reasoning_effort": "high",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_paw_config(config_path)
    # Unset -> no override; provider default applies.
    assert config.get_agent("generalist").reasoning_effort is None
    # Set per-agent.
    assert config.get_agent("applied-scientist").reasoning_effort == "high"
