"""Tests for generic CLI execution tool."""

import sys

from errand.tools.cli import run_cli


class FakeReplyTarget:
    def __init__(self, approved: bool):
        self.approved = approved
        self.requests = []

    async def request_approval(self, **kwargs):
        self.requests.append(kwargs)
        return self.approved


async def test_run_cli_success():
    result = await run_cli(
        f"{sys.executable} -c \"print('hello from cli')\"",
        timeout_seconds=5,
    )

    assert "Exit code: 0" in result
    assert "hello from cli" in result


async def test_run_cli_timeout():
    result = await run_cli(
        f"{sys.executable} -c \"import time; time.sleep(5)\"",
        timeout_seconds=1,
    )

    assert "timed out" in result


async def test_run_cli_requests_approval():
    reply_to = FakeReplyTarget(approved=True)
    result = await run_cli(
        f"{sys.executable} -c \"print('approved')\"",
        timeout_seconds=5,
        _context={"reply_to": reply_to, "agent_id": "generalist"},
    )

    assert reply_to.requests
    assert "approved" in result


async def test_run_cli_skips_when_denied():
    reply_to = FakeReplyTarget(approved=False)
    result = await run_cli(
        f"{sys.executable} -c \"print('should not run')\"",
        timeout_seconds=5,
        _context={"reply_to": reply_to, "agent_id": "generalist"},
    )

    assert reply_to.requests
    assert "not approved" in result
    assert "STDOUT:" not in result
