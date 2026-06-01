import asyncio
import contextlib

import uvicorn

from errand.interfaces.web.interface import WebInterface
from errand.contracts.interfaces import UserMessage
from errand.runtime.app import ErrandApp


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

    app = object.__new__(ErrandApp)
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


async def test_reset_command_archives_current_session() -> None:
    class Manager:
        def __init__(self):
            self.calls = []

        async def archive(self, session_id, start_new=False, *, agent_id=None):
            self.calls.append((session_id, start_new, agent_id))

    reply = _Reply()
    manager = Manager()
    app = object.__new__(ErrandApp)
    app.session_manager = manager
    app.config = type("Config", (), {"default_agent": "generalist"})()

    await app.handle_user_message(
        UserMessage("s1", "/new", "test", reply)
    )

    assert manager.calls == [("s1", True, "generalist")]
    assert reply.messages == ["Started a new conversation."]


async def test_reload_command_reloads_cached_prompts() -> None:
    class Manager:
        def __init__(self):
            self.get_args = None
            self.reload_args = None

        def get(self, session_id, *, agent_id=None, delegation_depth=0):
            self.get_args = (session_id, agent_id, delegation_depth)

        def reload_prompt_resources(self, *, soul=True, profile=True):
            self.reload_args = (soul, profile)
            return 2

    reply = _Reply()
    manager = Manager()
    app = object.__new__(ErrandApp)
    app.session_manager = manager
    app.config = type("Config", (), {"default_agent": "generalist"})()

    await app.handle_user_message(
        UserMessage("s1", "/reload soul", "test", reply)
    )

    assert manager.get_args == ("s1", "generalist", 0)
    assert manager.reload_args == (True, False)
    assert reply.messages == ["Reloaded soul for 2 active session(s)."]
