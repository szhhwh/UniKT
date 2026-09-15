"""File-backed log reading as rendered lines and WebSocket line streaming.

The raw ``.log`` is a PTY byte stream; :mod:`services.line_render` collapses it
to final rendered lines (rich colors, CR/erase-line/progress folded). This
module paginates those lines over HTTP and streams incremental line patches
over a WebSocket, keeping the ``data``/``done``/``error`` protocol the frontend
expects but operating on lines instead of raw bytes.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import WebSocket

from services.line_render import LineRenderCache

# Cap on the initial WS alignment patch so a bogus from_line against a huge log
# can't force-render the whole file into one frame (HTTP endpoint caps at 5000).
_WS_INITIAL_LIMIT = 5000


def read_log_lines(
    path: Path, cache: LineRenderCache, offset: int = 0, limit: int | None = None
) -> dict:
    """Return ``{"lines": [...], "total": int}`` for ``offset``/``limit``."""
    lines, total = cache.get(path, offset, limit)
    return {"lines": lines, "total": total}


async def stream_log_lines(
    path: Path,
    websocket: WebSocket,
    cache: LineRenderCache,
    check_alive: Callable[[], bool] | None = None,
    from_line: int = 0,
) -> None:
    """Stream rendered log lines over a WebSocket as incremental patches.

    Sends an initial ``patch`` aligning ``[from_line, total)`` — usually empty,
    since the client pre-loads the tail via HTTP — then polls the cache for new
    rows or an active-row refresh, emitting a ``patch`` per change. Terminates
    with ``done`` once the source is no longer alive.
    """
    # Rendering is pure-Python pyte work (a cold cache replays the whole file
    # byte by byte); keep it off the event loop or one big log stalls every
    # other request. The cache is RLock-guarded, so this is safe next to the
    # HTTP threadpool callers.
    lines, total = await asyncio.to_thread(
        cache.get, path, from_line, _WS_INITIAL_LIMIT
    )
    await _send(
        websocket,
        {"type": "patch", "from_line": from_line, "total": total, "lines": lines},
    )
    prev_total = total
    prev_sig = await asyncio.to_thread(cache.tail_repr, path)

    if not check_alive or not await asyncio.to_thread(check_alive):
        await _send(websocket, {"type": "done", "final": True})
        await websocket.close()
        return

    while True:
        alive = await asyncio.to_thread(check_alive)
        # Drain before the alive check short-circuits so the final bytes written
        # just before exit are not lost.
        patch = await asyncio.to_thread(cache.diff, path, prev_total, prev_sig)
        if patch:
            start, plines, ptotal, psig = patch
            await _send(
                websocket,
                {"type": "patch", "from_line": start, "total": ptotal, "lines": plines},
            )
            prev_total, prev_sig = ptotal, psig
        if not alive:
            break
        await asyncio.sleep(0.3)

    await _send(websocket, {"type": "done", "final": True})
    await websocket.close()


async def _send(websocket: WebSocket, message: dict) -> None:
    await websocket.send_json(message)
