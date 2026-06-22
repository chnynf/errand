"""Tests for Discord fallback-channel delivery."""

from paw.interfaces.discord_interface import (
    DiscordInterface,
    _FALLBACK_CHANNEL_NAME,
)


class FakeChannel:
    def __init__(self, name: str):
        self.name = name
        self.sent: list[str] = []

    async def send(self, content: str) -> None:
        self.sent.append(content)


class FakeGuild:
    def __init__(self, text_channels=None):
        self.text_channels = list(text_channels or [])
        self.created: list[str] = []

    async def create_text_channel(self, name, topic=None, reason=None):
        channel = FakeChannel(name)
        self.text_channels.append(channel)
        self.created.append(name)
        return channel


class FakeClient:
    def __init__(self, guilds):
        self.guilds = guilds

    def get_channel(self, channel_id):
        return None


def _interface_with_client(client) -> DiscordInterface:
    interface = object.__new__(DiscordInterface)
    interface._client = client
    return interface


async def test_fallback_creates_channel_in_single_guild() -> None:
    guild = FakeGuild()
    interface = _interface_with_client(FakeClient([guild]))

    ok = await interface.deliver_fallback_message("hello", context_id=None)

    assert ok is True
    assert guild.created == [_FALLBACK_CHANNEL_NAME]
    assert guild.text_channels[0].sent == ["hello"]


async def test_fallback_reuses_existing_channel() -> None:
    existing = FakeChannel(_FALLBACK_CHANNEL_NAME)
    guild = FakeGuild([existing])
    interface = _interface_with_client(FakeClient([guild]))

    ok = await interface.deliver_fallback_message("hello", context_id=None)

    assert ok is True
    assert guild.created == []
    assert existing.sent == ["hello"]


async def test_fallback_ambiguous_multiple_guilds_returns_false() -> None:
    interface = _interface_with_client(FakeClient([FakeGuild(), FakeGuild()]))

    ok = await interface.deliver_fallback_message("hello", context_id=None)

    assert ok is False


async def test_fallback_no_guild_returns_false() -> None:
    interface = _interface_with_client(FakeClient([]))

    ok = await interface.deliver_fallback_message("hello", context_id=None)

    assert ok is False
