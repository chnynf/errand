"""Per-session state: cache, locking, memory persistence."""

__all__ = ["AgentSession", "SessionManager"]


def __getattr__(name: str):
    if name == "AgentSession":
        from swarm_agent.sessions.session import AgentSession

        return AgentSession
    if name == "SessionManager":
        from swarm_agent.sessions.manager import SessionManager

        return SessionManager
    raise AttributeError(name)
