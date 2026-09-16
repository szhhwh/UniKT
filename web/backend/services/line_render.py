"""Render ``.log`` byte streams into final display lines via a pyte terminal.

PTY output (rich ANSI, carriage-return progress refresh, erase-line) is fed
through an in-memory terminal emulator so historical bytes collapse to their
final rendered state instead of replaying every refresh frame. Frozen rows are
stored compactly as segment lists; the pyte screen retains only a small tail,
keeping memory bounded regardless of total log size.

Threading: the owning reader thread calls ``feed`` as bytes are appended to the
log; HTTP/WS request threads call ``get``/``diff``. Each state is guarded by its
own ``RLock``; state creation is guarded by the instance lock.
"""

import threading
from collections import OrderedDict
from pathlib import Path

import pyte
from pyte.screens import Char

# Virtual terminal geometry. The PTY winsize is fixed to match (see
# process_manager / preprocess_manager) so rich wraps exactly as the subprocess
# expected when it wrote. 80 keeps progress bars compact enough to fit a normal
# viewport without wrapping; raise it if you want wider tables/bars.
COLUMNS = 80
LINES = 24

# pyte keeps at most this many scrolled-off rows; anything older is rendered
# into the compact frozen list. A small tail keeps the per-Char buffer tiny
# while the frozen list carries the full history as lightweight segment dicts.
_TAIL_ROWS = LINES
_READ_CHUNK = 65536
# Tail window compared/sent on each diff so multi-line redraws (rich Live/tqdm
# refreshing several rows via cursor-up within one poll) are captured.
_DIFF_OVERLAP = 64
# Bound on cached tasks; the oldest is evicted (rebuilt on next access) so a
# long-running server doesn't leak rendered history for every viewed task.
_STATE_CAPACITY = 64


def _render_row(row: dict[int, Char]) -> list[dict]:
    """Collapse a pyte ``Char`` row into styled segment dicts.

    Adjacent chars sharing the same style merge into one segment; default
    attributes are omitted to shrink the payload; trailing whitespace trims.
    """
    # ``row`` is a dict-like {x: Char}; missing positions resolve to the
    # default Char, so materializing 0..COLUMNS-1 yields a full styled line.
    segments: list[dict] = []
    key: tuple | None = None
    for x in range(COLUMNS):
        ch = row[x]
        style = (
            ch.fg,
            ch.bg,
            ch.bold,
            ch.italics,
            ch.underscore,
            ch.strikethrough,
            ch.reverse,
        )
        if style != key:
            seg = {"t": ch.data}
            if ch.fg != "default":
                seg["fg"] = ch.fg
            if ch.bg != "default":
                seg["bg"] = ch.bg
            if ch.bold:
                seg["bold"] = True
            if ch.italics:
                seg["italic"] = True
            if ch.underscore:
                seg["underline"] = True
            if ch.strikethrough:
                seg["strike"] = True
            if ch.reverse:
                seg["reverse"] = True
            segments.append(seg)
            key = style
        else:
            segments[-1]["t"] += ch.data
    while segments:
        stripped = segments[-1]["t"].rstrip()
        if stripped:
            segments[-1]["t"] = stripped
            break
        segments.pop()
    return segments


class _State:
    """Per-log terminal emulator plus compact frozen history."""

    __slots__ = ("fed_bytes", "frozen", "lock", "screen", "stream")

    def __init__(self) -> None:
        # history large enough that nothing is silently dropped before harvest.
        self.screen = pyte.HistoryScreen(COLUMNS, LINES, history=10**9)
        self.stream = pyte.ByteStream(self.screen)
        self.frozen: list[list[dict]] = []
        self.fed_bytes = 0
        self.lock = threading.RLock()


class LineRenderCache:
    """Per-path terminal emulator cache producing rendered log lines."""

    def __init__(self) -> None:
        """Create an empty cache."""
        self._lock = threading.RLock()
        self._states: OrderedDict[Path, _State] = OrderedDict()

    def feed(self, log_path: Path) -> None:
        """Advance the renderer with bytes appended to ``log_path`` since last feed."""
        state = self._ensure(log_path)
        with state.lock:
            self._catch_up(state, log_path)

    def get(
        self, log_path: Path, offset: int, limit: int | None = None
    ) -> tuple[list[list[dict]], int]:
        """Return ``(page_lines, total)`` for ``offset``/``limit`` pagination.

        ``limit=None`` returns every line from ``offset`` to the end.
        """
        state = self._ensure(log_path)
        with state.lock:
            self._catch_up(state, log_path)
            total = self._total(state)
            if offset < 0:
                offset = 0
            if offset >= total or limit == 0:
                return [], total
            end = total if limit is None else min(offset + limit, total)
            return self._range(state, offset, end), total

    def diff(
        self, log_path: Path, prev_total: int, prev_sig: str
    ) -> tuple[int, list[list[dict]], int, str] | None:
        """Return a patch since the last sent state, or ``None`` if unchanged.

        ``prev_sig`` is the signature of the last ``_DIFF_OVERLAP`` tail rows
        (see :meth:`tail_repr`). A change covers both newly appended rows and
        multi-line redraws (e.g. rich Live/tqdm refreshing several rows via
        cursor-up within one poll), not just the single last row.
        """
        state = self._ensure(log_path)
        with state.lock:
            self._catch_up(state, log_path)
            total = self._total(state)
            if total == 0:
                return None
            tail_start = max(0, total - _DIFF_OVERLAP)
            sig = repr(self._range(state, tail_start, total))
            if total == prev_total and sig == prev_sig:
                return None
            # Send from the earlier of (prev_total, tail_start): all new rows
            # when total grew, plus the whole redraw window when only the tail
            # changed.
            start = max(0, min(prev_total, tail_start))
            return start, self._range(state, start, total), total, sig

    def tail_repr(self, log_path: Path) -> str:
        """Return the signature of the current tail window (for WS init)."""
        state = self._ensure(log_path)
        with state.lock:
            self._catch_up(state, log_path)
            total = self._total(state)
            return repr(self._range(state, max(0, total - _DIFF_OVERLAP), total))

    def evict(self, log_path: Path) -> None:
        """Drop the cached state (e.g. when the task and its log are deleted)."""
        with self._lock:
            self._states.pop(log_path, None)

    def _ensure(self, log_path: Path) -> _State:
        with self._lock:
            state = self._states.get(log_path)
            if state is None:
                state = _State()
                self._states[log_path] = state
                while len(self._states) > _STATE_CAPACITY:
                    self._states.popitem(last=False)
            else:
                self._states.move_to_end(log_path)
            return state

    def _catch_up(self, state: _State, log_path: Path) -> None:
        # Drive the emulator purely from the file via fed_bytes, so a running
        # task's reader and a cold-start first read share one append path and
        # can never double-feed the same bytes.
        size = log_path.stat().st_size if log_path.is_file() else 0
        if size < state.fed_bytes:
            # Log was truncated/rotated out from under us — rebuild from zero
            # instead of seeking into the middle of a new file and feeding
            # garbage fragments onto stale history.
            state.fed_bytes = 0
            state.frozen = []
            state.screen.reset()
        if size == state.fed_bytes:
            return
        with open(log_path, "rb") as f:
            f.seek(state.fed_bytes)
            while True:
                chunk = f.read(_READ_CHUNK)
                if not chunk:
                    break
                state.stream.feed(chunk)
                state.fed_bytes += len(chunk)
        self._harvest(state)

    def _harvest(self, state: _State) -> None:
        top = state.screen.history.top
        while len(top) > _TAIL_ROWS:
            state.frozen.append(_render_row(top.popleft()))

    def _total(self, state: _State) -> int:
        # Nothing fed yet (empty/missing log) → no lines; cursor.y starts at 0
        # and would otherwise overcount by one phantom row.
        if state.fed_bytes == 0:
            return 0
        # Lines below the cursor were never written; exclude them.
        return (
            len(state.frozen)
            + len(state.screen.history.top)
            + (state.screen.cursor.y + 1)
        )

    def _line_at(self, state: _State, idx: int) -> list[dict]:
        nf = len(state.frozen)
        if idx < nf:
            return state.frozen[idx]
        nt = len(state.screen.history.top)
        if idx < nf + nt:
            return _render_row(state.screen.history.top[idx - nf])
        buf_idx = idx - nf - nt
        if 0 <= buf_idx <= state.screen.cursor.y:
            return _render_row(state.screen.buffer[buf_idx])
        return []

    def _range(self, state: _State, start: int, end: int) -> list[list[dict]]:
        return [self._line_at(state, i) for i in range(start, end)]
