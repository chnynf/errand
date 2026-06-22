"""CLI interface for Paw."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paw.contracts.interfaces import UserMessage

if TYPE_CHECKING:
    from paw.runtime.app import PawApp


@dataclass(frozen=True)
class ConsoleReplyTarget:
    """Reply target that writes to stdout."""

    async def send(self, message: str) -> None:
        print(f"Shell: {message}")
        print("-" * 50)

    async def send_progress(self, message: str) -> None:
        print(f"  ⋯ {message}")

    async def request_approval(
        self,
        *,
        title: str,
        details: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Ask for approval in the terminal."""
        print("\nApproval requested")
        print(f"{title}")
        print(details)
        print("-" * 50)
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(input, "Approve? [y/N]: "),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            print("Approval timed out.")
            return False
        return answer.strip().lower() in {"y", "yes", "approve", "approved"}


class CliInterface:
    """Interactive terminal interface."""

    name = "cli"

    def __init__(self, app: "PawApp", debug: bool = False):
        self._app = app
        self._debug = debug
        self._stopped = asyncio.Event()
        self._session_id = "cli_session"

    async def start(self) -> None:
        """Read terminal input until exit/quit."""
        print("Initializing Paw CLI...")
        if self._debug:
            print("Debug mode enabled.")
        print("Paw CLI Ready. Type 'exit' to quit.")
        print("-" * 50)

        while not self._stopped.is_set():
            try:
                user_input = (await asyncio.to_thread(input, "You: ")).strip()
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit"}:
                    await self.stop()
                    break
                if user_input.lower() == "archive":
                    print("Archiving session...")
                    await self._app.archive_session(self._session_id, start_new=True)
                    print("Session archived.")
                    continue

                await self._app.handle_user_message(
                    UserMessage(
                        session_id=self._session_id,
                        text=user_input,
                        source=self.name,
                        reply_to=ConsoleReplyTarget(),
                    )
                )
            except KeyboardInterrupt:
                await self.stop()
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    async def stop(self) -> None:
        """Stop the CLI loop."""
        self._stopped.set()
