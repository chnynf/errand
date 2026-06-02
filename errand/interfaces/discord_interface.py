"""Discord interface for Errand."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import discord
from aiohttp import ClientConnectorError, ClientOSError

from errand.contracts.interfaces import UserMessage

if TYPE_CHECKING:
    from errand.runtime.app import ErrandApp

_DISCORD_MAX_LEN = 2000
_MAPPING_FILE = Path(__file__).resolve().parent / "discord_session_mapping.json"
_DISCORD_RETRY_SECONDS = 5
_DISCORD_MAX_RETRY_SECONDS = 60
_FALLBACK_CHANNEL_NAME = "fallback-messages"


def _proxy_from_env() -> str | None:
    """Return a proxy URL from standard env vars, if configured."""
    return (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )


class DiscordReplyTarget:
    """Reply target for a Discord text channel."""

    def __init__(self, channel: discord.TextChannel, requester_id: int | None = None):
        self.channel = channel
        self.requester_id = requester_id
        self._status_message: discord.Message | None = None

    async def send(self, message: str) -> None:
        if self._status_message is not None:
            try:
                await self._status_message.delete()
            except discord.DiscordException:
                pass
            self._status_message = None
        for i in range(0, len(message), _DISCORD_MAX_LEN):
            await self.channel.send(message[i : i + _DISCORD_MAX_LEN])

    async def send_progress(self, message: str) -> None:
        content = f"*{message}*"
        if self._status_message is None:
            try:
                self._status_message = await self.channel.send(content)
            except discord.DiscordException:
                pass
        else:
            try:
                await self._status_message.edit(content=content)
            except discord.DiscordException:
                pass

    async def request_approval(
        self,
        *,
        title: str,
        details: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Ask for approval with Discord buttons."""
        view = ApprovalView(requester_id=self.requester_id, timeout=timeout_seconds)
        content = f"**Approval requested: {title}**\n\n{details}"
        if len(content) > _DISCORD_MAX_LEN:
            content = content[: _DISCORD_MAX_LEN - 20].rstrip() + "\n...[truncated]"
        message = await self.channel.send(content, view=view)
        approved = await view.wait_for_decision()
        for child in view.children:
            child.disabled = True
        try:
            await message.edit(view=view)
        except discord.DiscordException:
            pass
        return approved


class ApprovalView(discord.ui.View):
    """Discord approval buttons for a pending action."""

    def __init__(self, requester_id: int | None, timeout: float):
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self._decision: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.requester_id is None or interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the user who requested this action can approve it.",
            ephemeral=True,
        )
        return False

    async def wait_for_decision(self) -> bool:
        try:
            return await asyncio.wait_for(self._decision, timeout=self.timeout)
        except asyncio.TimeoutError:
            if not self._decision.done():
                self._decision.set_result(False)
            return False

    async def on_timeout(self) -> None:
        if not self._decision.done():
            self._decision.set_result(False)

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success)
    async def approve(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        if not self._decision.done():
            self._decision.set_result(True)
        await interaction.response.send_message("Approved.", ephemeral=True)
        self.stop()

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.danger)
    async def deny(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        if not self._decision.done():
            self._decision.set_result(False)
        await interaction.response.send_message("Denied.", ephemeral=True)
        self.stop()


class DiscordClient(discord.Client):
    """discord.py client that delegates application work to DiscordInterface."""

    def __init__(self, interface: "DiscordInterface", debug: bool = False):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        super().__init__(intents=intents, proxy=_proxy_from_env())
        self._interface = interface
        self._debug = debug

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("Ready to chat in all channels!")
        if self._debug:
            print("Debug mode enabled.")

    async def on_message(self, message) -> None:
        if message.author == self.user:
            return
        await self._interface.handle_discord_message(message)

    async def on_guild_channel_delete(self, channel) -> None:
        await self._interface.handle_channel_delete(channel)


class DiscordInterface:
    """Discord listener and delivery adapter."""

    name = "discord"

    def __init__(self, app: "ErrandApp", debug: bool = False):
        self._app = app
        self._debug = debug
        self._client = DiscordClient(self, debug=debug)
        self._mapping: dict[str, dict] = {}
        self._load_mapping()

    def is_configured(self) -> bool:
        """Whether the required Discord token exists."""
        return bool(os.getenv("DISCORD_TOKEN"))

    async def start(self) -> None:
        """Start the Discord client and block until it stops."""
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            raise ValueError("DISCORD_TOKEN not found in environment variables.")

        delay = _DISCORD_RETRY_SECONDS
        while True:
            try:
                await self._client.start(token, reconnect=True)
                return
            except discord.LoginFailure:
                raise
            except (ClientConnectorError, ClientOSError, TimeoutError, OSError) as e:
                print(f"Discord connection failed: {e}. Retrying in {delay}s.")
                await self.stop()
                await asyncio.sleep(delay)
                self._client = DiscordClient(self, debug=self._debug)
                delay = min(delay * 2, _DISCORD_MAX_RETRY_SECONDS)

    async def stop(self) -> None:
        """Stop the Discord client."""
        if not self._client.is_closed():
            await self._client.close()

    async def handle_discord_message(self, message) -> None:
        """Normalize and route a Discord message."""
        if self._debug:
            print(
                f"[DEBUG] Received message: {message.content!r} "
                f"from {message.author} in {message.channel} (ID: {message.channel.id})"
            )

        if not isinstance(message.channel, discord.TextChannel):
            return

        session_id = self._session_id_for_channel(message.channel.id)
        guild_id = message.guild.id if message.guild else None
        if guild_id and not session_id.startswith("scheduled:"):
            self._set_session_channel(str(message.channel.id), message.channel.id, guild_id)

        content = message.content.replace(f"<@{self._client.user.id}>", "").strip()
        if not content:
            content = "Hello"

        async with message.channel.typing():
            await self._app.handle_user_message(
                UserMessage(
                    session_id=session_id,
                    text=content,
                    source=self.name,
                    reply_to=DiscordReplyTarget(
                        message.channel,
                        requester_id=message.author.id,
                    ),
                )
            )

    async def handle_channel_delete(self, channel) -> None:
        """Archive session state when a Discord text channel is deleted."""
        if not isinstance(channel, discord.TextChannel):
            return

        session_id = self._session_id_for_channel(channel.id)
        try:
            await self._app.archive_session(session_id, start_new=False)
            if self._debug:
                print(f"[DEBUG] Archived session {session_id}")
        except Exception as e:
            print(f"Error archiving session for channel {channel.id}: {e}")

    async def deliver_scheduled_result(
        self,
        task_session_id: str,
        message: str,
        context_id: str | None = None,
    ) -> bool:
        """Deliver scheduled job results to Discord."""
        entry = self._mapping.get(str(task_session_id))

        if not entry:
            if not context_id:
                print(
                    f"Scheduler delivery: no mapping for {task_session_id} "
                    "and no context_id provided"
                )
                return False
            entry = self._guild_entry_from_context(context_id)
            if not entry:
                return False

        target_channel_id = entry.get("channel_id")
        if target_channel_id:
            channel = self._client.get_channel(target_channel_id)
            if isinstance(channel, discord.TextChannel):
                await DiscordReplyTarget(channel).send(message)
                return True

        guild = self._client.get_guild(entry.get("guild_id"))
        if not guild:
            print(
                f"Scheduler delivery: guild {entry.get('guild_id')} "
                f"not found for session {task_session_id}"
            )
            return False

        try:
            channel_name = f"scheduled-{str(task_session_id).replace(':', '-')[:20]}"
            channel_name = "".join(c for c in channel_name if c.isalnum() or c in "-").lower()
            channel = discord.utils.get(guild.text_channels, name=channel_name)
            if channel is None:
                channel = await guild.create_text_channel(
                    channel_name,
                    topic=f"Scheduled task result: {task_session_id}",
                    reason="Scheduled task delivery",
                )

            self._mapping[str(task_session_id)] = {
                "channel_id": channel.id,
                "guild_id": guild.id,
            }
            self._save_mapping()
            await DiscordReplyTarget(channel).send(message)
            return True
        except discord.DiscordException as e:
            print(f"Scheduler delivery error: {e}")
            return False

    async def deliver_fallback_message(
        self,
        message: str,
        context_id: str | None = None,
    ) -> bool:
        """Post a message to the standing fallback channel.

        Used when a message could not be delivered through its original
        interface. Picks a guild from the failed origin context when possible,
        otherwise the sole connected guild, then sends to a text channel named
        ``fallback-messages`` (creating it if needed).
        """
        guild = self._fallback_guild(context_id)
        if guild is None:
            return False

        channel = await self._get_or_create_fallback_channel(guild)
        if channel is None:
            return False

        try:
            await DiscordReplyTarget(channel).send(message)
            return True
        except discord.DiscordException as e:
            print(f"Fallback delivery error: {e}")
            return False

    def _fallback_guild(self, context_id: str | None):
        """Resolve the guild for fallback delivery, or None if ambiguous."""
        guild = self._guild_from_context(context_id)
        if guild is not None:
            return guild

        guilds = list(self._client.guilds)
        if len(guilds) == 1:
            return guilds[0]
        if not guilds:
            print("Fallback delivery: bot is not connected to any guild.")
            return None
        print(
            "Fallback delivery: multiple guilds connected and context "
            f"{context_id!r} did not identify one; cannot choose a guild."
        )
        return None

    def _guild_from_context(self, context_id: str | None):
        """Return the guild for a Discord channel-id context, else None (quiet)."""
        try:
            channel = self._client.get_channel(int(context_id))
        except (TypeError, ValueError):
            return None
        if isinstance(channel, discord.TextChannel) and channel.guild:
            return channel.guild
        return None

    async def _get_or_create_fallback_channel(self, guild):
        channel = discord.utils.get(guild.text_channels, name=_FALLBACK_CHANNEL_NAME)
        if channel is not None:
            return channel
        try:
            return await guild.create_text_channel(
                _FALLBACK_CHANNEL_NAME,
                topic="Errand messages that could not be delivered to their original channel.",
                reason="Errand fallback delivery",
            )
        except discord.DiscordException as e:
            print(f"Fallback delivery: failed to create #{_FALLBACK_CHANNEL_NAME}: {e}")
            return None

    def _session_id_for_channel(self, channel_id: int) -> str:
        for session_id, entry in self._mapping.items():
            if (
                session_id.startswith("scheduled:")
                and isinstance(entry, dict)
                and entry.get("channel_id") == channel_id
            ):
                return session_id
        return str(channel_id)

    def _guild_entry_from_context(self, context_id: str) -> dict | None:
        try:
            context_channel = self._client.get_channel(int(context_id))
        except ValueError:
            print(f"Scheduler delivery: invalid context_id {context_id}")
            return None
        if isinstance(context_channel, discord.TextChannel) and context_channel.guild:
            return {"guild_id": context_channel.guild.id}
        print(f"Scheduler delivery: context channel {context_id} not found or not text/guild")
        return None

    def _load_mapping(self) -> None:
        if not _MAPPING_FILE.exists():
            return
        try:
            with open(_MAPPING_FILE, "r", encoding="utf-8") as f:
                self._mapping = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._mapping = {}

    def _save_mapping(self) -> None:
        _MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(self._mapping, f, indent=2)

    def _set_session_channel(self, session_id: str, channel_id: int, guild_id: int) -> None:
        self._mapping[str(session_id)] = {"channel_id": channel_id, "guild_id": guild_id}
        self._save_mapping()
