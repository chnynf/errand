"""Per-session agent execution."""

from errand.agent_loop import AgentLoop


class AgentSession:
    """One persisted agent conversation.

    Each chat, scheduled job, or CLI conversation gets an AgentSession.
    The session owns the agent loop and the memory backing it.
    """

    def __init__(
        self,
        session_id: str,
        debug: bool = False,
        *,
        agent_id: str | None = None,
        config=None,
        delegation_depth: int = 0,
    ):
        self.session_id = session_id
        self.agent_id = agent_id
        self._loop = AgentLoop(
            session_id=session_id,
            debug=debug,
            agent_id=agent_id,
            config=config,
            delegation_depth=delegation_depth,
        )

    async def process(self, text: str, metadata: dict | None = None) -> str:
        """Process one user/scheduled input and return the final response."""
        return await self._loop.process_input(text, metadata=metadata)

    def add_session_note(self, note: str) -> None:
        self._loop.add_session_note(note)

    def last_activity_at(self) -> float | None:
        return self._loop.last_activity_at()

    def reload_prompt_resources(self, *, soul: bool = True, profile: bool = True) -> None:
        self._loop.reload_prompt_resources(soul=soul, profile=profile)

    async def archive(self, start_new: bool = False) -> None:
        """Archive the persisted session file."""
        await self._loop.memory.archive_session(start_new=start_new)

    async def shutdown(self) -> None:
        """Persist session state."""
        await self._loop.shutdown()
