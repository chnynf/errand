"""Web dashboard interface for Errand."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from errand.runtime.app import ErrandApp

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7861


class WebInterface:
    """Serves the Errand dashboard over HTTP."""

    name = "web"

    def __init__(self, app: "ErrandApp", *, host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT):
        self._app = app
        self._host = host
        self._port = port
        self._server: Any | None = None

    def is_configured(self) -> bool:
        return True

    async def start(self) -> None:
        try:
            import uvicorn
        except ImportError:
            print("Web interface requires uvicorn. Run: pip install uvicorn[standard]")
            return

        from errand.interfaces.web.api import build_api

        fastapi_app = build_api(self._app)
        config = uvicorn.Config(
            fastapi_app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server.capture_signals = contextlib.nullcontext
        self._server = server
        print(f"Web dashboard: http://{self._host}:{self._port}")
        try:
            await server.serve()
        except asyncio.CancelledError:
            server.should_exit = True
            if server.started:
                await server.shutdown()
            raise
        finally:
            if self._server is server:
                self._server = None

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
