"""SessionManager: cache AgentSession instances and serialize work per session."""

import asyncio

from errand.config import ErrandConfig, load_errand_config
from errand.sessions.session import AgentSession


class SessionManager:
    """Owns AgentSession instances and serializes work per session."""

    def __init__(self, debug: bool = False, *, config: ErrandConfig | None = None):
        self._debug = debug
        self._config = config or load_errand_config()
        self._sessions: dict[str, AgentSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        delegation_depth: int = 0,
    ) -> AgentSession:
        """Get or create a session."""
        resolved_agent_id = self._config.get_agent(agent_id).id
        cache_key = f"{resolved_agent_id}:{session_id}"
        if cache_key not in self._sessions:
            self._sessions[cache_key] = AgentSession(
                session_id=session_id,
                debug=self._debug,
                agent_id=resolved_agent_id,
                config=self._config,
                delegation_depth=delegation_depth,
            )
        return self._sessions[cache_key]

    async def process(
        self,
        session_id: str,
        text: str,
        metadata: dict | None = None,
        agent_id: str | None = None,
        delegation_depth: int = 0,
    ) -> str:
        """Process text with per-session serialization."""
        resolved_agent_id = self._config.get_agent(agent_id).id
        lock_key = f"{resolved_agent_id}:{session_id}"
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            return await self.get(
                session_id,
                agent_id=resolved_agent_id,
                delegation_depth=delegation_depth,
            ).process(text, metadata=metadata)

    async def archive(self, session_id: str, start_new: bool = False) -> None:
        """Archive a session and remove its cached runner."""
        resolved_agent_id = self._config.default_agent
        cache_key = f"{resolved_agent_id}:{session_id}"
        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            session = self.get(session_id, agent_id=resolved_agent_id)
            await session.archive(start_new=start_new)
            if not start_new:
                self._sessions.pop(cache_key, None)

    async def shutdown(self) -> None:
        """Persist all cached sessions."""
        await asyncio.gather(
            *(session.shutdown() for session in self._sessions.values()),
            return_exceptions=True,
        )
