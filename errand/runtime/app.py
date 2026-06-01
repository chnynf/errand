"""ErrandApp: process lifecycle and component wiring."""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from errand.config import ErrandConfig, load_errand_config
from errand.contracts.interfaces import ErrandInterface, UserMessage
from errand.runtime.debug import set_debug
from errand.scheduler.service import SchedulerService


class ErrandApp:
    """Owns the process lifecycle, interfaces, sessions, and scheduler."""

    def __init__(
        self,
        *,
        config: ErrandConfig | None = None,
        debug: bool = False,
        interfaces: Optional[Iterable[str]] = None,
    ):
        load_dotenv()
        set_debug(debug)
        self.debug = debug
        self.config = config or load_errand_config()

        # SessionManager is imported lazily so ``--help`` doesn't pay for
        # the full agent/brain/litellm import chain at module load time.
        from errand.sessions import SessionManager

        self.session_manager: Any = SessionManager(debug=debug, config=self.config)
        self._interface_names = list(interfaces) if interfaces else self.config.enabled_interfaces()
        self._interfaces: list[ErrandInterface] = []
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
        print(f"Errand runtime started. Interfaces: {interface_names}{scheduler_suffix}")

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
        await message.reply_to.send(response)

    async def _handle_control_command(self, message: UserMessage, agent_id: str) -> bool:
        command = message.text.strip().lower()
        if command in {"/new", "/reset", "archive"}:
            await self.session_manager.archive(
                message.session_id,
                start_new=True,
                agent_id=agent_id,
            )
            await message.reply_to.send("Started a new conversation.")
            return True

        if not command.startswith("/reload"):
            return False

        parts = command.split()
        target = parts[1] if len(parts) > 1 else "prompts"
        if target in {"prompts", "all"}:
            soul = profile = True
            label = "runtime instructions"
        elif target == "soul":
            soul, profile, label = True, False, "soul"
        elif target in {"index", "profile"}:
            soul, profile, label = False, True, "profile"
        else:
            await message.reply_to.send("Usage: /reload [soul|index|profile|prompts]")
            return True

        self.session_manager.get(message.session_id, agent_id=agent_id)
        count = self.session_manager.reload_prompt_resources(soul=soul, profile=profile)
        await message.reply_to.send(f"Reloaded {label} for {count} active session(s).")
        return True

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
        """Ask interfaces to deliver a scheduled result."""
        for interface in self._interfaces:
            deliver = getattr(interface, "deliver_scheduled_result", None)
            if not deliver:
                continue
            handled = await deliver(task_session_id, message, context_id=context_id)
            if handled:
                return True
        return False

    def _build_interfaces(self, names: list[str]) -> list[ErrandInterface]:
        interfaces: list[ErrandInterface] = []
        for name in names:
            if name == "discord":
                from errand.interfaces.discord_interface import DiscordInterface

                interface = DiscordInterface(self, debug=self.debug)
                if interface.is_configured():
                    interfaces.append(interface)
                else:
                    print("Skipping Discord interface: DISCORD_TOKEN is not set.")
            elif name == "cli":
                from errand.interfaces.cli import CliInterface

                interfaces.append(CliInterface(self, debug=self.debug))
            elif name == "web":
                from errand.interfaces.web import WebInterface

                interfaces.append(WebInterface(self))
            elif name == "wechat":
                from errand.interfaces.wechat_interface import WeChatInterface

                interface = WeChatInterface(self, debug=self.debug)
                if interface.is_configured():
                    interfaces.append(interface)
                else:
                    print(
                        "Skipping WeChat interface: no credentials found.\n"
                        "Run: wechat-clawbot-cc setup"
                    )
            else:
                print(f"Skipping unknown interface: {name}")
        return interfaces


async def run_errand(
    *,
    debug: bool = False,
    interfaces: Optional[Iterable[str]] = None,
) -> None:
    """Convenience wrapper used by the command-line entrypoint."""
    app = ErrandApp(debug=debug, interfaces=interfaces)
    await app.start()
