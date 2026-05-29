"""WeChat iLink (ClawBot) interface for Errand.

Uses Tencent's official iLink Bot API directly via httpx.
No third-party bot libraries — every HTTP call is visible in this file.

All network traffic goes to ilinkai.weixin.qq.com (Tencent's servers).
This is the same data path as using WeChat itself.

HTTP calls are made with synchronous httpx.Client run in asyncio.to_thread
to avoid the anyio/uvicorn event-loop backend conflict that arises when the
web interface is also enabled.

First run: errand prints a QR code; scan it in WeChat → Me → Settings →
Plugins → ClawBot. Credentials are saved to wechat_creds.json so subsequent
starts skip the login.

Each WeChat user gets an isolated session: wechat:{from_user_id}
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from errand.contracts.interfaces import UserMessage

if TYPE_CHECKING:
    from errand.runtime.app import ErrandApp

# ---------------------------------------------------------------------------
# iLink API
# ---------------------------------------------------------------------------
_ILINK_BASE = "https://ilinkai.weixin.qq.com"

_MSG_TYPE_USER = 1
_MSG_TYPE_BOT = 2
_MSG_STATE_FINISH = 2
_ITEM_TYPE_TEXT = 1

# ---------------------------------------------------------------------------
# Runtime constants
# ---------------------------------------------------------------------------
_POLL_TIMEOUT = 40.0
_RETRY_DELAY = 5.0
_MAX_LEN = 2000

_CREDS_FILE = Path(__file__).parent / "wechat_creds.json"
_STATE_FILE = Path(__file__).parent / "wechat_state.json"

_APPROVE_KEYWORDS = frozenset({"同意", "批准", "确认", "approve", "yes", "y", "ok"})
_DENY_KEYWORDS = frozenset({"拒绝", "否决", "取消", "deny", "no", "n", "cancel"})


# ---------------------------------------------------------------------------
# HTTP helpers — synchronous inside asyncio.to_thread to avoid anyio conflict
# ---------------------------------------------------------------------------

def _base_headers(uin: str = "") -> dict:
    """Headers required by every iLink request (authenticated or not)."""
    return {
        "Content-Type": "application/json",
        "iLink-App-Id": "bot",
        "iLink-App-ClientVersion": "65536",  # encodes version 1.0.0
        "X-WECHAT-UIN": uin or base64.b64encode(os.urandom(4)).decode(),
    }


def _auth_headers(creds: dict) -> dict:
    """Build HTTP headers for authenticated iLink requests."""
    h = _base_headers(creds.get("uin", ""))
    h["AuthorizationType"] = "ilink_bot_token"
    h["Authorization"] = f"Bearer {creds['token']}"
    return h


def _sync_get(url: str, params: dict, headers: dict, timeout: float) -> dict:
    with httpx.Client() as c:
        r = c.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()


def _sync_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    with httpx.Client() as c:
        r = c.post(url, json=body, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()


async def _get(url: str, params: dict, headers: dict, timeout: float = 15.0) -> dict:
    return await asyncio.to_thread(_sync_get, url, params, headers, timeout)


async def _post(url: str, body: dict, headers: dict, timeout: float = 30.0) -> dict:
    return await asyncio.to_thread(_sync_post, url, body, headers, timeout)


# ---------------------------------------------------------------------------
# QR display
# ---------------------------------------------------------------------------

_QR_FILE = Path(__file__).parent / "wechat_qr.png"


def _display_qr(qr_token: str, qr_page_url: str = "") -> None:
    """Show the QR code the user must scan in WeChat.

    qr_page_url is a liteapp.weixin.qq.com URL whose page displays the QR code.
    We open it in the browser (macOS) and also generate a local PNG fallback.
    """
    import subprocess

    sep = "=" * 60

    if qr_page_url:
        print(f"\n{sep}")
        print("WeChat Bot: scan the QR code to log in")
        print(f"{sep}")
        print(f"URL: {qr_page_url}")
        print("\nOpening in browser — scan the QR code shown on that page with WeChat.")
        print("WeChat → Me → Settings → Plugins → ClawBot → Scan")
        print(f"{sep}\n")
        subprocess.Popen(
            ["open", qr_page_url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Also generate a local PNG (encode the URL itself as a QR code)
        try:
            import qrcode as qrc  # type: ignore

            qr = qrc.QRCode(border=1)
            qr.add_data(qr_page_url)
            qr.make(fit=True)
            img = qr.make_image()
            img.save(_QR_FILE)
            print(f"(Local QR PNG also saved → {_QR_FILE})\n")
        except Exception:
            pass
        return

    # Fallback: ASCII QR from the token
    try:
        import qrcode as qrc  # type: ignore

        qr = qrc.QRCode(border=1)
        qr.add_data(qr_token)
        qr.make(fit=True)
        print()
        qr.print_ascii(invert=True)
        print()
    except ImportError:
        print(f"\nQR token: {qr_token}")
        print("(pip install qrcode[pil] for terminal display)\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str) -> list[str]:
    return [text[i : i + _MAX_LEN] for i in range(0, len(text), _MAX_LEN)] or [""]


def _extract_text(msg: dict) -> str:
    for item in msg.get("item_list") or []:
        if item.get("type") == _ITEM_TYPE_TEXT:
            return (item.get("text_item") or {}).get("text", "").strip()
    return ""


# ---------------------------------------------------------------------------
# ReplyTarget
# ---------------------------------------------------------------------------

class WeChatReplyTarget:
    """Sends agent responses back to a specific WeChat user via iLink API."""

    def __init__(
        self,
        creds: dict,
        *,
        from_user_id: str,
        context_token: str,
        session_id: str,
        pending_approvals: dict,
    ):
        self._creds = creds
        self._from_user_id = from_user_id
        self._context_token = context_token
        self._session_id = session_id
        self._pending_approvals = pending_approvals

    async def send(self, message: str) -> None:
        for chunk in _chunk(message):
            payload = {
                "msg": {
                    "to_user_id": self._from_user_id,
                    "client_id": str(uuid.uuid4()),
                    "message_type": _MSG_TYPE_BOT,
                    "message_state": _MSG_STATE_FINISH,
                    "item_list": [
                        {"type": _ITEM_TYPE_TEXT, "text_item": {"text": chunk}}
                    ],
                    "context_token": self._context_token,
                }
            }
            try:
                await _post(
                    f"{self._creds['base_url']}/ilink/bot/sendmessage",
                    payload,
                    _auth_headers(self._creds),
                    timeout=10.0,
                )
            except Exception as e:
                print(f"WeChat send error: {e}")

    async def send_progress(self, message: str) -> None:
        """Show typing indicator (best-effort)."""
        try:
            data = await _post(
                f"{self._creds['base_url']}/ilink/bot/getconfig",
                {"context_token": self._context_token},
                _auth_headers(self._creds),
                timeout=5.0,
            )
            ticket = data.get("typing_ticket")
            if ticket:
                await _post(
                    f"{self._creds['base_url']}/ilink/bot/sendtyping",
                    {"typing_ticket": ticket, "context_token": self._context_token},
                    _auth_headers(self._creds),
                    timeout=5.0,
                )
        except Exception:
            pass

    async def request_approval(
        self,
        *,
        title: str,
        details: str,
        timeout_seconds: int = 300,
    ) -> bool:
        await self.send(f"【待确认】{title}\n\n{details}\n\n请回复「同意」或「拒绝」")
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_approvals[self._session_id] = future
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=float(timeout_seconds)
            )
        except asyncio.TimeoutError:
            await self.send("确认超时，操作已取消。")
            return False
        finally:
            self._pending_approvals.pop(self._session_id, None)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class WeChatInterface:
    """WeChat iLink Bot — long-poll receive, thread-safe sync httpx send."""

    name = "wechat"

    def __init__(self, app: "ErrandApp", debug: bool = False):
        self._app = app
        self._debug = debug
        self._creds: dict | None = None
        self._cursor: str = ""
        self._stopped = asyncio.Event()
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}

    def is_configured(self) -> bool:
        return True  # login runs on first start if no credentials exist

    async def start(self) -> None:
        self._creds = _load_creds()
        if not self._creds:
            self._creds = await self._login()
        self._cursor = _load_cursor()
        print("WeChat Bot connected. Listening for messages...")
        await self._poll_loop()

    async def stop(self) -> None:
        self._stopped.set()

    # ------------------------------------------------------------------
    # Login (QR code flow)
    # ------------------------------------------------------------------

    async def _login(self) -> dict:
        """Interactive QR login. Retries automatically if QR expires."""
        print("WeChat Bot: no saved credentials — starting QR login...")

        uin = base64.b64encode(os.urandom(4)).decode()
        login_headers = _base_headers(uin)

        while True:
            data = await _get(
                f"{_ILINK_BASE}/ilink/bot/get_bot_qrcode",
                {"bot_type": "3"},
                login_headers,
                timeout=15.0,
            )
            qr_token: str = data.get("qrcode", "")
            qr_img_url: str = data.get("qrcode_img_content", "")

            _display_qr(qr_token, qr_img_url)
            print("Waiting for scan (you have ~3 minutes)", end="", flush=True)

            expired = False
            while not expired:
                await asyncio.sleep(2)
                print(".", end="", flush=True)
                try:
                    status = await _get(
                        f"{_ILINK_BASE}/ilink/bot/get_qrcode_status",
                        {"qrcode": qr_token},
                        login_headers,
                        timeout=60.0,   # long enough for any server-side wait
                    )
                except Exception as e:
                    print(f"\nWeChat: status poll error ({e}), retrying...")
                    continue

                if status.get("bot_token"):
                    print(" ✓")
                    bot_id: str = status.get("ilink_bot_id", "")
                    creds = {
                        "token": status["bot_token"],
                        "base_url": (
                            status.get("bot_base_url")
                            or status.get("baseurl")
                            or _ILINK_BASE
                        ),
                        "uin": uin,
                        "bot_id": bot_id,
                    }
                    _save_creds(creds)
                    print("WeChat Bot: login successful. Credentials saved.")
                    if bot_id:
                        print(f"Bot WeChat ID: {bot_id}")
                        print("Share this ID with family so they can add the bot.")
                    return creds

                if status.get("status") == "expired":
                    print("\nQR code expired — generating a new one...")
                    expired = True

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Long-poll /ilink/bot/getupdates and dispatch each user message."""
        while not self._stopped.is_set():
            try:
                data = await _post(
                    f"{self._creds['base_url']}/ilink/bot/getupdates",
                    {
                        "get_updates_buf": self._cursor,
                        "base_info": {"channel_version": "1.0.2"},
                    },
                    _auth_headers(self._creds),
                    timeout=_POLL_TIMEOUT,
                )

                # Detect expired/invalid token and re-login automatically
                ret = data.get("ret", 0)
                if ret in (-14, -1):  # session timeout / auth error
                    print(f"WeChat: token expired (ret={ret}), re-authenticating...")
                    _CREDS_FILE.unlink(missing_ok=True)
                    _STATE_FILE.unlink(missing_ok=True)
                    self._cursor = ""
                    self._creds = await self._login()
                    continue

                new_cursor = data.get("get_updates_buf")
                if new_cursor:
                    self._cursor = new_cursor
                    _save_cursor(self._cursor)

                for msg in data.get("msgs") or []:
                    if msg.get("message_type") != _MSG_TYPE_USER:
                        continue
                    asyncio.create_task(self._handle_message(msg))

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._stopped.is_set():
                    print(f"WeChat poll error: {e}. Retrying in {_RETRY_DELAY}s.")
                    await asyncio.sleep(_RETRY_DELAY)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict) -> None:
        from_user_id: str = msg.get("from_user_id", "")
        context_token: str = msg.get("context_token", "")
        session_id = f"wechat:{from_user_id}"

        text = _extract_text(msg)
        if not text:
            return

        if self._debug:
            print(f"[DEBUG] WeChat message: {text!r} from {from_user_id}")

        reply_target = WeChatReplyTarget(
            self._creds,
            from_user_id=from_user_id,
            context_token=context_token,
            session_id=session_id,
            pending_approvals=self._pending_approvals,
        )

        if await self._try_resolve_approval(session_id, text, reply_target):
            return

        await self._app.handle_user_message(
            UserMessage(
                session_id=session_id,
                text=text,
                source=self.name,
                reply_to=reply_target,
            )
        )

    async def _try_resolve_approval(
        self,
        session_id: str,
        content: str,
        reply_target: WeChatReplyTarget,
    ) -> bool:
        future = self._pending_approvals.get(session_id)
        if not future or future.done():
            return False
        normalized = content.strip().lower()
        if normalized in _APPROVE_KEYWORDS:
            future.set_result(True)
            return True
        if normalized in _DENY_KEYWORDS:
            future.set_result(False)
            return True
        await reply_target.send("请回复「同意」或「拒绝」以完成确认。")
        return True


# ---------------------------------------------------------------------------
# Credential and cursor persistence
# ---------------------------------------------------------------------------

def _load_creds() -> dict | None:
    token = os.getenv("WECHAT_BOT_TOKEN")
    if token:
        return {
            "token": token,
            "base_url": os.getenv("WECHAT_BOT_BASE_URL", _ILINK_BASE),
            "uin": base64.b64encode(os.urandom(4)).decode(),
        }
    if _CREDS_FILE.exists():
        try:
            return json.loads(_CREDS_FILE.read_text())
        except Exception as e:
            print(f"WeChat: failed to read credentials: {e}")
    return None


def _save_creds(creds: dict) -> None:
    _CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CREDS_FILE.write_text(json.dumps(creds, indent=2))


def _load_cursor() -> str:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text()).get("cursor", "")
        except Exception:
            pass
    return ""


def _save_cursor(cursor: str) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({"cursor": cursor}))
