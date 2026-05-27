"""FastAPI app for the Swarm web dashboard."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

if TYPE_CHECKING:
    from swarm_agent.runtime.app import SwarmApp

_SESSIONS_DIR = Path(__file__).resolve().parents[3] / "swarm_agent" / "sessions" / "_data"
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.json"
_DASHBOARD_HTML = Path(__file__).resolve().parent / "dashboard.html"

COMPONENTS = [
    {
        "id": "config",
        "label": "Config",
        "description": "Loads config.json and provides SwarmConfig to all components.",
        "connects_to": ["brain", "agent_loop", "sessions", "scheduler", "interfaces"],
    },
    {
        "id": "contracts",
        "label": "Contracts",
        "description": "Shared types: ToolCall, BrainDecision, UserMessage, ReplyTarget.",
        "connects_to": ["brain", "agent_loop", "tools", "interfaces"],
    },
    {
        "id": "interfaces",
        "label": "Interfaces",
        "description": "Discord, CLI, Web — normalize external input into UserMessage.",
        "connects_to": ["runtime"],
    },
    {
        "id": "runtime",
        "label": "Runtime",
        "description": "SwarmApp wires components, owns process lifecycle.",
        "connects_to": ["sessions", "scheduler"],
    },
    {
        "id": "sessions",
        "label": "Sessions",
        "description": "Persists conversation history and token usage per session.",
        "connects_to": ["agent_loop"],
    },
    {
        "id": "agent_loop",
        "label": "Agent Loop",
        "description": "Text in → text out. Orchestrates brain + tools + memory.",
        "connects_to": ["brain", "tools"],
    },
    {
        "id": "brain",
        "label": "Brain",
        "description": "Prompt + tool defs → BrainDecision. Provider-agnostic via LiteLLM.",
        "connects_to": [],
    },
    {
        "id": "tools",
        "label": "Tools",
        "description": "Everything the brain can invoke: functions, file access, delegation.",
        "connects_to": ["agent_loop"],
    },
    {
        "id": "scheduler",
        "label": "Scheduler",
        "description": "Manages cron jobs and fires them through the runtime.",
        "connects_to": ["runtime"],
    },
]


def build_api(swarm_app: "SwarmApp") -> FastAPI:
    app = FastAPI(title="Swarm Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        html = _DASHBOARD_HTML.read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/components")
    async def get_components() -> JSONResponse:
        return JSONResponse({"components": COMPONENTS})

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        try:
            raw = _CONFIG_PATH.read_text(encoding="utf-8")
            return JSONResponse(json.loads(raw))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.patch("/api/config")
    async def patch_config(body: dict[str, Any]) -> JSONResponse:
        try:
            raw = _CONFIG_PATH.read_text(encoding="utf-8")
            config = json.loads(raw)
            _deep_merge(config, body)
            _CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")
            return JSONResponse({"ok": True})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sessions")
    async def get_sessions() -> JSONResponse:
        sessions = _load_sessions()
        return JSONResponse({"sessions": sessions})

    @app.get("/api/tokens")
    async def get_tokens() -> JSONResponse:
        sessions = _load_sessions()
        daily: dict[str, dict[str, float]] = {}
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for s in sessions:
            summary = s.get("token_summary", {})
            day = s.get("day", "unknown")
            if day not in daily:
                daily[day] = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
            daily[day]["input_tokens"] += summary.get("input_tokens", 0)
            daily[day]["output_tokens"] += summary.get("output_tokens", 0)
            daily[day]["total_cost"] += summary.get("total_cost", 0.0)
            total_input += summary.get("input_tokens", 0)
            total_output += summary.get("output_tokens", 0)
            total_cost += summary.get("total_cost", 0.0)

        return JSONResponse({
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 4),
            "by_day": daily,
        })

    return app


def _load_sessions() -> list[dict[str, Any]]:
    if not _SESSIONS_DIR.exists():
        return []
    results = []
    for path in sorted(_SESSIONS_DIR.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
            day = time.strftime("%Y-%m-%d", time.localtime(mtime))
            results.append({
                "session_id": data.get("session_id", path.stem),
                "agent_id": data.get("metadata", {}).get("agent_id", "unknown"),
                "created_at": data.get("created_at"),
                "day": day,
                "message_count": len([
                    e for e in data.get("history", [])
                    if e.get("role") == "user"
                ]),
                "token_summary": data.get("token_summary", {}),
            })
        except Exception:
            continue
    return results


def _deep_merge(base: dict, patch: dict) -> None:
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
