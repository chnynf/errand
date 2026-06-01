import time

from errand.config import AgentSpec, ErrandConfig, SessionConfig
from errand.sessions.manager import SessionManager
from errand.sessions.memory import IDLE_BOUNDARY_NOTE


def test_idle_boundary_marks_next_turn_context(monkeypatch, tmp_path):
    monkeypatch.setattr("errand.sessions.memory._SESSION_DIR", tmp_path)
    config = ErrandConfig(
        model_strategy=["dummy"],
        sessions=SessionConfig(idle_boundary_hours=1),
        agents={
            "default": AgentSpec(
                id="default",
                model_strategy=["dummy"],
            )
        },
    )
    manager = SessionManager(config=config)
    session = manager.get("s1")
    session._loop.memory.add_history("user", "old topic")
    session._loop.memory.data["history"][-1]["timestamp"] = time.time() - 7200

    manager._mark_idle_boundary(session)
    context, _ = session._loop.memory.get_formatted_context()

    assert f"SESSION NOTE:\n{IDLE_BOUNDARY_NOTE}" in context
    assert "USER: old topic" in context
