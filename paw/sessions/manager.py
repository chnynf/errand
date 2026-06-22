"""SessionManager: cache AgentSession instances and serialize work per session."""

import asyncio
import time

from paw.config import PawConfig, load_paw_config
from paw.sessions.session import AgentSession
from paw.sessions.memory import IDLE_BOUNDARY_NOTE, RELOAD_BOUNDARY_NOTE


class SessionManager:
    """Owns AgentSession instances and serializes work per session."""

    def __init__(self, debug: bool = False, *, config: PawConfig | None = None):
        self._debug = debug
        self._config = config or load_paw_config()
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
            session = self.get(
                session_id,
                agent_id=resolved_agent_id,
                delegation_depth=delegation_depth,
            )
            self._mark_idle_boundary(session)
            return await session.process(text, metadata=metadata)

    def _mark_idle_boundary(self, session: AgentSession) -> None:
        hours = self._config.sessions.idle_boundary_hours
        last_activity = session.last_activity_at()
        if hours and last_activity and time.time() - last_activity > hours * 3600:
            session.add_session_note(IDLE_BOUNDARY_NOTE)

    async def archive(
        self,
        session_id: str,
        start_new: bool = False,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Archive a session and remove its cached runner."""
        resolved_agent_id = self._config.get_agent(agent_id).id
        cache_key = f"{resolved_agent_id}:{session_id}"
        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            session = self.get(session_id, agent_id=resolved_agent_id)
            await session.archive(start_new=start_new)
            if not start_new:
                self._sessions.pop(cache_key, None)

    def reload_prompt_resources(
        self,
        *,
        soul: bool = True,
        profile: bool = True,
        note: str = RELOAD_BOUNDARY_NOTE,
    ) -> int:
        """Clear prompt caches for cached sessions and mark a reload boundary."""
        for session in self._sessions.values():
            session.reload_prompt_resources(soul=soul, profile=profile)
            session.add_session_note(note)
        return len(self._sessions)

    async def shutdown(self) -> None:
        """Persist all cached sessions."""
        await asyncio.gather(
            *(session.shutdown() for session in self._sessions.values()),
            return_exceptions=True,
        )
