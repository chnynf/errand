"""PawApp: process lifecycle and component wiring."""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from paw.config import PawConfig, load_paw_config
from paw.contracts.interfaces import PawInterface, UserMessage
from paw.runtime.control import (
    NEW_SESSION_MESSAGE,
    is_new_session_command,
    reload_message,
)
from paw.runtime.debug import set_debug
from paw.scheduler.service import SchedulerService


class PawApp:
    """Owns the process lifecycle, interfaces, sessions, and scheduler."""

    def __init__(
        self,
        *,
        config: PawConfig | None = None,
        debug: bool = False,
        interfaces: Optional[Iterable[str]] = None,
    ):
        load_dotenv()
        set_debug(debug)
        self.debug = debug
        self.config = config or load_paw_config()

        # SessionManager is imported lazily so ``--help`` doesn't pay for
        # the full agent/brain/litellm import chain at module load time.
        from paw.sessions import SessionManager

        self.session_manager: Any = SessionManager(debug=debug, config=self.config)
        self._interface_names = list(interfaces) if interfaces else self.config.enabled_interfaces()
        self._interfaces: list[PawInterface] = []
        self._tasks: list[asyncio.Task] = []
        self._scheduler: SchedulerService | None = None

    async def start(self) -> None:
        """Start enabled interfaces and services, then block until stopped."""
        if not self._interface_names:
            raise ValueError(
                "No interfaces enabled. Set config.json interfaces.*.enabled or pass --interface."
            )

        self._interfaces = self._build_interfaces(self._interface_names)
        if not self._interfaces:
            raise ValueError("No runnable interfaces found.")

        interface_tasks = [
            asyncio.create_task(interface.start(), name=f"interface:{interface.name}")
            for interface in self._interfaces
        ]
        self._tasks = list(interface_tasks)

        if self.config.scheduler.enabled:
            self._scheduler = SchedulerService(
                self,
                poll_seconds=self.config.scheduler.poll_seconds,
            )
            self._tasks.append(asyncio.create_task(self._scheduler.start(), name="scheduler"))

        interface_names = ", ".join(interface.name for interface in self._interfaces)
        scheduler_suffix = ", scheduler" if self.config.scheduler.enabled else ""
        print(f"Paw runtime started. Interfaces: {interface_names}{scheduler_suffix}")

        try:
            # Interfaces define runtime lifetime. Background services like the
            # scheduler are stopped once the last interface exits.
            await asyncio.gather(*interface_tasks)
        except asyncio.CancelledError:
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop services and persist sessions."""
        for interface in self._interfaces:
            try:
                await interface.stop()
            except Exception as e:
                print(f"Error stopping {interface.name} interface: {e}")

        if self._scheduler:
            await self._scheduler.stop()

        pending = [task for task in self._tasks if not task.done()]
        if pending:
            _, pending = await asyncio.wait(pending, timeout=2.0)
            for task in pending:
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        await self.session_manager.shutdown()

    async def handle_user_message(self, message: UserMessage) -> None:
        """Process a normalized user message and reply through its target."""
        metadata = dict(message.metadata or {})
        metadata["_reply_to"] = message.reply_to
        metadata["_source"] = message.source
        agent_id = metadata.get("agent_id") or self.config.default_agent
        if await self._handle_control_command(message, agent_id):
            return
        response = await self.session_manager.process(
            message.session_id,
            message.text,
            metadata=metadata,
            agent_id=agent_id,
        )
        await self._send_final(
            message.reply_to,
            response,
            source=message.source,
            context_id=message.session_id,
        )

    async def _handle_control_command(
        self,
        message: UserMessage,
        agent_id: str,
    ) -> bool:
        command = message.text.strip().lower()
        if is_new_session_command(command):
            await self.session_manager.archive(
                message.session_id,
                start_new=True,
                agent_id=agent_id,
            )
            await self._send_control_reply(message, NEW_SESSION_MESSAGE)
            return True

        if not command.startswith("/reload"):
            return False

        reload_args = self.parse_reload_command(command)
        if reload_args is None:
            await message.reply_to.send("Usage: /reload [soul|index|profile|prompts]")
            return True

        soul, profile, target = reload_args
        self.reload_prompt_resources(message.session_id, agent_id, soul=soul, profile=profile)
        await self._send_control_reply(message, reload_message(target))
        return True

    async def _send_control_reply(self, message: UserMessage, text: str) -> None:
        """Send a static control-command confirmation straight to the user.

        Control commands never invoke the model -- this sends a fixed,
        prefixed status line instead of running a turn just to acknowledge.
        """
        await self._send_final(
            message.reply_to,
            text,
            source=message.source,
            context_id=message.session_id,
        )

    def reload_prompt_resources(
        self,
        session_id: str,
        agent_id: str,
        *,
        soul: bool = True,
        profile: bool = True,
    ) -> None:
        self.session_manager.get(session_id, agent_id=agent_id)
        self.session_manager.reload_prompt_resources(soul=soul, profile=profile)

    @staticmethod
    def parse_reload_command(command: str) -> tuple[bool, bool, str] | None:
        parts = command.split()
        target = parts[1] if len(parts) > 1 else "prompts"
        if target in {"prompts", "all"}:
            return True, True, "runtime instructions"
        elif target == "soul":
            return True, False, "soul"
        elif target in {"index", "profile"}:
            return False, True, "profile"
        return None

    async def archive_session(self, session_id: str, start_new: bool = False) -> None:
        """Archive a session by ID."""
        await self.session_manager.archive(session_id, start_new=start_new)

    async def process_scheduled_job(self, session_id: str, text: str, name: str) -> str:
        """Run a scheduled job prompt through its own session."""
        return await self.session_manager.process(
            session_id,
            text,
            metadata={"is_scheduled_task": True, "job_name": name},
            agent_id=self.config.default_agent,
        )

    async def deliver_scheduled_result(
        self,
        task_session_id: str,
        message: str,
        context_id: str | None = None,
    ) -> bool:
        """Deliver a scheduled result to its origin, falling back on failure."""
        for interface in self._interfaces:
            deliver = getattr(interface, "deliver_scheduled_result", None)
            if not deliver:
                continue
            handled = await deliver(task_session_id, message, context_id=context_id)
            if handled:
                return True
        return await self._send_fallback(
            self._source_label(context_id),
            message,
            context_id=context_id,
        )

    async def _send_final(
        self,
        reply_to,
        message: str,
        *,
        source: str,
        context_id: str | None = None,
    ) -> bool:
        """Send a final user-visible message, falling back to Discord on failure."""
        try:
            await reply_to.send(message)
            return True
        except Exception as e:
            print(f"Send failed via {source}: {e}")
            return await self._send_fallback(source, message, context_id=context_id)

    async def _send_fallback(
        self,
        source: str,
        message: str,
        *,
        context_id: str | None = None,
    ) -> bool:
        """Route an undeliverable message to a fallback-capable interface."""
        text = (
            f"I tried to reach you on {source}, but couldn't send the message. "
            f"Here's the message: {message}"
        )
        for interface in self._interfaces:
            deliver = getattr(interface, "deliver_fallback_message", None)
            if not deliver:
                continue
            try:
                if await deliver(text, context_id=context_id):
                    return True
            except Exception as e:
                print(f"Fallback delivery error via {interface.name}: {e}")
        print(f"Fallback delivery failed: no interface delivered message from {source}")
        return False

    @staticmethod
    def _source_label(context_id: str | None) -> str:
        """Best-effort human label for the origin a message could not reach."""
        if not context_id:
            return "your original channel"
        if ":" in context_id:
            return context_id.split(":", 1)[0]
        if context_id == "cli_session":
            return "cli"
        if str(context_id).isdigit():
            return "discord"
        return context_id

    def _build_interfaces(self, names: list[str]) -> list[PawInterface]:
        def _load(name: str):
            if name == "discord":
                from paw.interfaces.discord_interface import DiscordInterface
                return DiscordInterface(self, debug=self.debug), "Discord", "DISCORD_TOKEN is not set."
            if name == "cli":
                from paw.interfaces.cli import CliInterface
                return CliInterface(self, debug=self.debug), "CLI", None
            if name == "web":
                from paw.interfaces.web import WebInterface
                return WebInterface(self), "Web", None
            if name == "wechat":
                from paw.interfaces.wechat_interface import WeChatInterface
                return (
                    WeChatInterface(self, debug=self.debug),
                    "WeChat",
                    "no credentials found.\nRun: wechat-clawbot-cc setup",
                )
            return None, None, None

        interfaces: list[PawInterface] = []
        for name in names:
            interface, label, skip_reason = _load(name)
            if interface is None:
                print(f"Skipping unknown interface: {name}")
                continue
            if skip_reason and not interface.is_configured():
                print(f"Skipping {label} interface: {skip_reason}")
                continue
            interfaces.append(interface)
        return interfaces


async def run_paw(
    *,
    debug: bool = False,
    interfaces: Optional[Iterable[str]] = None,
) -> None:
    """Convenience wrapper used by the command-line entrypoint."""
    app = PawApp(debug=debug, interfaces=interfaces)
    await app.start()
