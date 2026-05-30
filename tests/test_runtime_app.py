import asyncio
import contextlib

import uvicorn

from errand.interfaces.web.interface import WebInterface
from errand.runtime.app import ErrandApp


class _SessionManager:
    def __init__(self, calls: list[str]):
        self._calls = calls

    async def shutdown(self) -> None:
        self._calls.append("session.shutdown")


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
