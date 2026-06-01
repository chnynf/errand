import asyncio

from errand.interfaces.wechat_interface import (
    WeChatInterface,
    _ITEM_TYPE_TEXT,
    _MSG_TYPE_USER,
)


class _App:
    def __init__(self):
        self.messages = []

    async def handle_user_message(self, message):
        self.messages.append(message)


class _Reply:
    async def send(self, message: str) -> None:
        pass


async def test_wechat_control_command_bypasses_pending_approval():
    app = _App()
    interface = WeChatInterface(app)
    interface._creds = {"base_url": "https://example.com", "token": "token"}
    future = asyncio.get_running_loop().create_future()
    interface._pending_approvals["wechat:u1"] = future

    await interface._handle_message(
        {
            "message_type": _MSG_TYPE_USER,
            "from_user_id": "u1",
            "context_token": "ctx",
            "item_list": [{"type": _ITEM_TYPE_TEXT, "text_item": {"text": "/new"}}],
        }
    )

    assert [message.text for message in app.messages] == ["/new"]
    assert "wechat:u1" not in interface._pending_approvals
    assert future.result() is False


async def test_wechat_reload_command_bypasses_helper_reply_and_prompts():
    app = _App()
    interface = WeChatInterface(app)
    interface._creds = {"base_url": "https://example.com", "token": "token"}

    await interface._handle_message(
        {
            "message_type": _MSG_TYPE_USER,
            "from_user_id": "u1",
            "context_token": "ctx",
            "item_list": [{"type": _ITEM_TYPE_TEXT, "text_item": {"text": "/reload soul"}}],
        }
    )

    assert [message.text for message in app.messages] == ["/reload soul"]


async def test_completed_wechat_approval_does_not_intercept_message():
    interface = WeChatInterface(_App())
    future = asyncio.get_running_loop().create_future()
    future.set_result(False)
    interface._pending_approvals["wechat:u1"] = future

    handled = await interface._try_resolve_approval("wechat:u1", "hello", _Reply())

    assert handled is False
