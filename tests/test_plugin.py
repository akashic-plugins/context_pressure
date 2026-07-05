from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.lifecycle.types import AfterStepCtx
from plugin import (
    ContextPressureStopModule,
    _CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS,
)


@pytest.mark.asyncio
async def test_context_pressure_stops_when_threshold_exceeded() -> None:
    ctx = AfterStepCtx(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        iteration=0,
        context_tokens_estimate=_CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS + 1,
        tools_called=(),
        partial_reply="",
        tools_used_so_far=(),
        tool_chain_partial=(),
        partial_thinking=None,
        has_more=True,
    )
    frame = SimpleNamespace(slots={"step:ctx": ctx})
    await ContextPressureStopModule().run(frame)
    assert frame.slots["step:early_stop_reason"] == "context_pressure"


@pytest.mark.asyncio
async def test_context_pressure_ignores_safe_context_size() -> None:
    ctx = AfterStepCtx(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        iteration=0,
        context_tokens_estimate=128,
        tools_called=(),
        partial_reply="",
        tools_used_so_far=(),
        tool_chain_partial=(),
        partial_thinking=None,
        has_more=True,
    )
    frame = SimpleNamespace(slots={"step:ctx": ctx})
    await ContextPressureStopModule().run(frame)
    assert "step:early_stop_reason" not in frame.slots
