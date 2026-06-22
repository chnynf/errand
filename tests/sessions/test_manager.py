import time

from paw.config import AgentSpec, PawConfig, SessionConfig
from paw.sessions.manager import SessionManager
from paw.sessions.memory import IDLE_BOUNDARY_NOTE


def test_idle_boundary_marks_next_turn_context(monkeypatch, tmp_path):
    monkeypatch.setattr("paw.sessions.memory._SESSION_DIR", tmp_path)
    config = PawConfig(
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

    assert session._loop.memory.data["metadata"]["session_note"] == IDLE_BOUNDARY_NOTE
    assert session._loop.memory.build_history_messages() == [
        {"role": "user", "content": "old topic"}
    ]
