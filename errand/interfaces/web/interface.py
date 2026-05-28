"""Web dashboard interface for Errand."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

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
        self._server: asyncio.Task | None = None

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
        print(f"Web dashboard: http://{self._host}:{self._port}")
        await server.serve()

    async def stop(self) -> None:
        pass
