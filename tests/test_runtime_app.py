import asyncio
import contextlib

import uvicorn

from paw.interfaces.web.interface import WebInterface
from paw.contracts.interfaces import UserMessage
from paw.runtime.control import NEW_SESSION_MESSAGE, reload_message
from paw.runtime.app import PawApp


class _SessionManager:
    def __init__(self, calls: list[str]):
        self._calls = calls

    async def shutdown(self) -> None:
        self._calls.append("session.shutdown")


class _Reply:
    def __init__(self):
        self.messages: list[str] = []

    async def send(self, message: str) -> None:
        self.messages.append(message)

    async def send_progress(self, message: str) -> None:
        pass

    async def request_approval(self, **kwargs) -> bool:
        return True


class _Interface:
    name = "fake"

    def __init__(self, calls: list[str], stopped: asyncio.Event):
        self._calls = calls
        self._stopped = stopped

    async def start(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        self._calls.append("interface.stop")
        self._stopped.set()


async def test_app_stop_lets_interfaces_exit_before_cancelling_tasks() -> None:
    calls: list[str] = []
    stopped = asyncio.Event()
    interface = _Interface(calls, stopped)
    task = asyncio.create_task(interface.start())
    await asyncio.sleep(0)

    app = object.__new__(PawApp)
    app._interfaces = [interface]
    app._tasks = [task]
    app._scheduler = None
    app.session_manager = _SessionManager(calls)

    await app.stop()

    assert calls == ["interface.stop", "session.shutdown"]
    assert task.done()
    assert not task.cancelled()


async def test_web_interface_disables_uvicorn_signal_capture(monkeypatch) -> None:
    class FakeConfig:
        def __init__(self, app, **kwargs):
            self.app = app
            self.kwargs = kwargs

    class FakeServer:
        instances: list["FakeServer"] = []

        def __init__(self, config):
            self.config = config
            self.capture_signals = object()
            self.should_exit = False
            self.started = False
            self.shutdown_called = False
            FakeServer.instances.append(self)

        async def serve(self) -> None:
            self.started = True
            while not self.should_exit:
                await asyncio.sleep(0)

        async def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)

    interface = WebInterface(object())
    task = asyncio.create_task(interface.start())
    await asyncio.sleep(0)

    server = FakeServer.instances[0]
    assert server.capture_signals is contextlib.nullcontext

    await interface.stop()
    await asyncio.wait_for(task, timeout=1)

    assert server.should_exit


async def test_reset_command_archives_and_replies_without_model() -> None:
    class Manager:
        def __init__(self):
            self.calls = []
            self.process_calls = []

        async def archive(self, session_id, start_new=False, *, agent_id=None):
            self.calls.append((session_id, start_new, agent_id))

        async def process(self, session_id, text, metadata=None, agent_id=None):
            self.process_calls.append((session_id, text, metadata, agent_id))
            return "should not be called"

    reply = _Reply()
    manager = Manager()
    app = object.__new__(PawApp)
    app.session_manager = manager
    app.config = type("Config", (), {"default_agent": "generalist"})()

    await app.handle_user_message(
        UserMessage("s1", "/new", "test", reply)
    )

    assert manager.calls == [("s1", True, "generalist")]
    # No model turn: the control command replies with a static, prefixed line.
    assert manager.process_calls == []
    assert reply.messages == [NEW_SESSION_MESSAGE]


async def test_scheduled_job_archives_then_runs_in_fresh_session() -> None:
    """Each scheduled fire starts clean: archive (start_new) before processing,
    so the prior run's history is not replayed/re-billed on the next fire."""
    order: list[str] = []

    class Manager:
        async def archive(self, session_id, start_new=False, *, agent_id=None):
            order.append(("archive", session_id, start_new, agent_id))

        async def process(self, session_id, text, metadata=None, agent_id=None):
            order.append(("process", session_id, metadata.get("is_scheduled_task"), agent_id))
            return "ran"

    app = object.__new__(PawApp)
    app.session_manager = Manager()
    app.config = type("Config", (), {"default_agent": "generalist"})()

    result = await app.process_scheduled_job("scheduled:job-x", "do the task", name="nightly")

    assert result == "ran"
    # Archive must happen first, and with start_new=True under the same agent.
    assert order == [
        ("archive", "scheduled:job-x", True, "generalist"),
        ("process", "scheduled:job-x", True, "generalist"),
    ]


class _RaisingReply:
    async def send(self, message: str) -> None:
        raise RuntimeError("send failed")

    async def send_progress(self, message: str) -> None:
        pass

    async def request_approval(self, **kwargs) -> bool:
        return True


class _FallbackInterface:
    name = "discord"

    def __init__(self, *, scheduled_result: bool = True, fallback_result: bool = True):
        self.scheduled_calls: list[tuple] = []
        self.fallback_calls: list[tuple] = []
        self._scheduled_result = scheduled_result
        self._fallback_result = fallback_result

    async def deliver_scheduled_result(self, task_session_id, message, context_id=None):
        self.scheduled_calls.append((task_session_id, message, context_id))
        return self._scheduled_result

    async def deliver_fallback_message(self, message, context_id=None):
        self.fallback_calls.append((message, context_id))
        return self._fallback_result


def _app_with_interfaces(interfaces) -> PawApp:
    app = object.__new__(PawApp)
    app._interfaces = interfaces
    return app


async def test_send_final_success_skips_fallback() -> None:
    fb = _FallbackInterface()
    app = _app_with_interfaces([fb])
    reply = _Reply()

    result = await app._send_final(reply, "hi", source="wechat", context_id="wechat:1")

    assert result is True
    assert reply.messages == ["hi"]
    assert fb.fallback_calls == []


async def test_send_final_failure_routes_to_fallback() -> None:
    fb = _FallbackInterface()
    app = _app_with_interfaces([fb])

    result = await app._send_final(
        _RaisingReply(), "hi", source="wechat", context_id="wechat:1"
    )

    assert result is True
    assert len(fb.fallback_calls) == 1
    text, context_id = fb.fallback_calls[0]
    assert "I tried to reach you on wechat" in text
    assert "hi" in text
    assert context_id == "wechat:1"


async def test_scheduled_origin_success_skips_fallback() -> None:
    fb = _FallbackInterface(scheduled_result=True)
    app = _app_with_interfaces([fb])

    result = await app.deliver_scheduled_result("scheduled:job-1", "do it", context_id="123")

    assert result is True
    assert fb.fallback_calls == []


async def test_scheduled_origin_failure_routes_to_fallback() -> None:
    fb = _FallbackInterface(scheduled_result=False)
    app = _app_with_interfaces([fb])

    result = await app.deliver_scheduled_result(
        "scheduled:job-1", "do it", context_id="wechat:42"
    )

    assert result is True
    assert len(fb.fallback_calls) == 1
    text, _ = fb.fallback_calls[0]
    assert "I tried to reach you on wechat" in text
    assert "do it" in text


async def test_fallback_failure_returns_false() -> None:
    fb = _FallbackInterface(fallback_result=False)
    app = _app_with_interfaces([fb])

    result = await app._send_final(
        _RaisingReply(), "hi", source="wechat", context_id="wechat:1"
    )

    assert result is False


async def test_reload_command_reloads_cached_prompts() -> None:
    class Manager:
        def __init__(self):
            self.get_args = None
            self.reload_args = None
            self.process_calls = []

        def get(self, session_id, *, agent_id=None, delegation_depth=0):
            self.get_args = (session_id, agent_id, delegation_depth)

        def reload_prompt_resources(self, *, soul=True, profile=True):
            self.reload_args = (soul, profile)
            return 2

        async def process(self, session_id, text, metadata=None, agent_id=None):
            self.process_calls.append((session_id, text, metadata, agent_id))
            return "Reloaded."

    reply = _Reply()
    manager = Manager()
    app = object.__new__(PawApp)
    app.session_manager = manager
    app.config = type("Config", (), {"default_agent": "generalist"})()

    await app.handle_user_message(
        UserMessage("s1", "/reload soul", "test", reply)
    )

    assert manager.get_args == ("s1", "generalist", 0)
    assert manager.reload_args == (True, False)
    # No model turn: the control command replies with a static, prefixed line.
    assert manager.process_calls == []
    assert reply.messages == [reload_message("soul")]
