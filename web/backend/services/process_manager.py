"""Process manager — experiment task lifecycle with a multi-GPU scheduler.

One background thread owns the queue, the running-process table, and the PTY
file descriptors. Each GPU is a scheduling lane with ``gpu_slots`` concurrent
slots; tasks request either a specific GPU or auto-assignment and are
dispatched once a lane has a free slot. Per-lane occupancy is derived from the
DB (rows whose status is running/stopping/interrupted, grouped by
``gpu_assigned``) rather than a hand counter, so lifecycle code never touches
slot accounting. With no GPUs the scheduler collapses to a single CPU lane.
Task output is appended to per-task ``.log`` files. All status writes go
through the CAS state machine in ``task_state.transition``.
"""

import contextlib
import fcntl
import json
import logging
import os
import pty
import signal
import struct
import subprocess
import termios
import threading
import time
from collections import deque
from datetime import datetime

import psutil
from config import PROJECT_ROOT, TASK_LOGS_DIR
from database import SessionLocal
from models import Task

from services.cli_builder import build_param_flags
from services.gpu_monitor import GpuMonitor
from services.line_render import LineRenderCache
from services.pid_utils import pid_reused
from services.python_env import PythonEnvManager
from services.schema_extractor import SchemaExtractor
from services.task_state import transition

logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages experiment task subprocesses with a multi-GPU execution queue.

    Args:
        env_manager: PythonEnvManager used to resolve task commands.
        gpu_monitor: GpuMonitor used to detect the number of GPU lanes.
    """

    def __init__(
        self,
        env_manager: PythonEnvManager,
        gpu_monitor: GpuMonitor,
        schema_extractor: SchemaExtractor,
        line_cache: LineRenderCache,
    ):
        """Initialize the manager and start the background scheduler thread."""
        self._env_manager = env_manager
        self._gpu_monitor = gpu_monitor
        self._schema_extractor = schema_extractor
        self._line_cache = line_cache
        self._queue: deque[tuple[int, int | None]] = deque()
        self._running: dict[int, subprocess.Popen] = {}
        self._master_fds: dict[int, int] = {}
        self._readers: dict[int, threading.Thread] = {}
        self._recover_monitors: dict[int, threading.Thread] = {}
        self._gpu_slots_capacity = 1
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def gpu_slots(self) -> int:
        """Concurrent task slots available on each GPU (or the CPU lane)."""
        return self._gpu_slots_capacity

    @gpu_slots.setter
    def gpu_slots(self, value: int) -> None:
        with self._lock:
            self._gpu_slots_capacity = max(1, value)
        self._wake.set()

    def get_queue(self) -> list[int]:
        """Return a copy of the ordered task ID queue."""
        with self._lock:
            return [tid for tid, _ in self._queue]

    def reorder_queue(self, task_ids: list[int]) -> None:
        """Move the given task IDs to the front of the queue in order."""
        with self._lock:
            by_id = dict(self._queue)
            front = set(task_ids)
            preserved = [(tid, req) for tid, req in self._queue if tid not in front]
            # Dedupe: task_ids is client-supplied, and a repeated id would put
            # the same task in the queue twice and spawn a second trainer.
            seen: set[int] = set()
            valid = []
            for tid in task_ids:
                if tid in by_id and tid not in seen:
                    seen.add(tid)
                    valid.append((tid, by_id[tid]))
            self._queue = deque(valid + preserved)

    def remove_from_queue(self, task_id: int) -> bool:
        """Remove a task from the queue; return True if it was present."""
        with self._lock:
            for idx, (tid, _) in enumerate(self._queue):
                if tid == task_id:
                    del self._queue[idx]
                    return True
            return False

    def launch_task(
        self,
        task_id: int,
        model_name: str,
        params: dict,
        env_id: str,
        custom_python_path: str | None = None,
    ) -> None:
        """Stamp the task row with resolved command/env and enqueue it."""
        with SessionLocal() as session:
            task = session.get(Task, task_id)
            if not task:
                return

            base_cmd = self._env_manager.resolve_command(env_id, custom_python_path)
            cmd = base_cmd + self._build_cli_args(model_name, params)
            env_type, env_name = env_id.split(":", 1)

            task.command = " ".join(cmd)
            task.model_name = model_name
            task.dataset_name = params.get("dataset", "")
            task.env_type = env_type
            task.env_name = env_name
            # _do_launch re-resolves the command from the row, so the custom
            # interpreter path must be persisted here as well.
            task.python_path = custom_python_path or ""
            # Search rows keep the default separators so the LIKE marker
            # matches; train rows are dumped compact so no nested param value
            # can forge the marker substring.
            separators = None if params.get("task_kind") == "optuna" else (",", ":")
            task.extra_params = json.dumps(params, separators=separators)
            session.commit()
            gpu_request = task.gpu_request

        with self._lock:
            self._queue.append((task_id, gpu_request))
        self._wake.set()

    def _loop(self) -> None:
        while not self._stopping:
            try:
                self._reap()
                self._launch_pending()
            except Exception:
                logger.exception("scheduler loop error")
            self._wake.wait(timeout=0.2)
            self._wake.clear()

    def _launch_pending(self) -> None:
        lanes = self._lanes()
        cap = self._gpu_slots_capacity
        usage = self._slot_usage()
        while not self._stopping:
            tid, assigned = self._pop_dispatchable(lanes, cap, usage)
            if tid is None:
                return
            self._do_launch(tid, assigned)

    def _gpu_count(self) -> int:
        """Return the stable GPU device count (0 if NVML is unavailable)."""
        return self._gpu_monitor.device_count

    def _lanes(self) -> list[int | None]:
        """Return the scheduling lanes: one per GPU, or a single CPU lane."""
        count = self._gpu_count()
        return list(range(count)) if count > 0 else [None]

    def _slot_usage(self) -> dict[int | None, int]:
        """Count tasks occupying each lane (running/stopping/interrupted)."""
        with SessionLocal() as session:
            rows = (
                session.query(Task.gpu_assigned)
                .filter(Task.status.in_(["running", "stopping", "interrupted"]))
                .all()
            )
        usage: dict[int | None, int] = {}
        for (gpu,) in rows:
            usage[gpu] = usage.get(gpu, 0) + 1
        return usage

    def _pick_lane(
        self,
        request: int | None,
        lanes: list[int | None],
        cap: int,
        usage: dict[int | None, int],
    ) -> tuple[bool, int | None]:
        """Resolve a task to a lane with a free slot.

        Returns ``(True, lane)`` when a dispatchable lane exists (``lane`` may be
        ``None`` for the CPU lane), or ``(False, None)`` when the task is
        blocked. Pinned requests target their GPU only; auto requests pick the
        least-loaded lane that still has capacity (ties favor the lower index,
        preserved by lane order).
        """
        if request is not None and request in lanes:
            return (usage.get(request, 0) < cap, request)
        best: int | None = None
        best_load = cap
        found = False
        for lane in lanes:
            load = usage.get(lane, 0)
            if load < cap and load < best_load:
                best = lane
                best_load = load
                found = True
        return (found, best)

    def _pop_dispatchable(
        self,
        lanes: list[int | None],
        cap: int,
        usage: dict[int | None, int],
    ) -> tuple[int | None, int | None]:
        """Pop the first queue task that can be dispatched now.

        ``lanes``/``cap``/``usage`` are computed once per scheduler pass by the
        caller; this only takes the lock for the in-memory queue scan (no I/O
        under the lock) and mutates ``usage`` in place so multiple pops in one
        pass account for just-dispatched tasks without re-querying. Scans in
        order, skipping tasks whose target lane is full so a blocked pinned task
        does not stall later dispatchable tasks.
        """
        with self._lock:
            for idx, (tid, request) in enumerate(self._queue):
                fallback = request is not None and request not in lanes
                ok, target = self._pick_lane(
                    None if fallback else request, lanes, cap, usage
                )
                if not ok:
                    continue
                if fallback:
                    logger.warning(
                        "task %s requested GPU %s but only %d lane(s) exist; "
                        "auto-assigned to lane %s",
                        tid,
                        request,
                        len(lanes),
                        target,
                    )
                del self._queue[idx]
                usage[target] = usage.get(target, 0) + 1
                return tid, target
            return None, None

    def _do_launch(self, task_id: int, assigned_gpu: int | None) -> bool:
        with SessionLocal() as session:
            task = session.get(Task, task_id)
            if not task:
                return False

            try:
                env_id = f"{task.env_type}:{task.env_name}"
                custom_python_path = task.python_path or None
                base_cmd = self._env_manager.resolve_command(env_id, custom_python_path)
                params = json.loads(task.extra_params or "{}")
                cmd = base_cmd + self._build_cli_args(task.model_name, params)
                env_type = task.env_type
            except Exception:
                logger.exception("command build failed for task %s", task_id)
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "failed",
                    finished_at=datetime.now(),
                )
                return False

            try:
                master_fd, slave_fd = pty.openpty()
            except OSError:
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "failed",
                    finished_at=datetime.now(),
                )
                return False

            try:
                # Matches the pyte emulator columns so rich wraps exactly as
                # rendered downstream; the frontend no longer resizes the PTY.
                winsize = struct.pack("HHHH", 24, 80, 0, 0)
                fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
                env = os.environ.copy()
                env["TERM"] = "xterm-256color"
                env["FORCE_COLOR"] = "1"
                if assigned_gpu is not None:
                    # Lanes are NVML indices (PCI bus order); CUDA defaults to
                    # FASTEST_FIRST, so without this the ordinal would select a
                    # different physical card than the scheduler accounted for.
                    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
                proc = subprocess.Popen(
                    cmd,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=str(PROJECT_ROOT),
                    start_new_session=True,
                    env=env,
                )
                os.close(slave_fd)
            except Exception:
                os.close(slave_fd)
                os.close(master_fd)
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "failed",
                    finished_at=datetime.now(),
                )
                return False

            if self._stopping:
                # Shutdown raced the launch: kill the fresh process instead of
                # registering it. The row stays pending, so the next backend
                # start re-queues it — no orphan, no double dispatch.
                self._kill_process_group(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Undead (uninterruptible sleep): mark failed so the next
                    # start does not re-dispatch alongside the survivor.
                    transition(
                        session,
                        Task,
                        task_id,
                        "pending",
                        "failed",
                        finished_at=datetime.now(),
                    )
                with contextlib.suppress(OSError):
                    os.close(master_fd)
                return False

            # Register the proc before the DB transition so a concurrent
            # stop_task/kill_task sees and can terminate it instead of racing
            # the transition and orphaning the subprocess.
            with self._lock:
                self._running[task_id] = proc
                self._master_fds[task_id] = master_fd

            extra: dict = {}
            if env_type == "custom" and custom_python_path:
                extra["python_path"] = custom_python_path
            try:
                claimed = transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "running",
                    pid=proc.pid,
                    started_at=datetime.now(),
                    gpu_assigned=assigned_gpu,
                    **extra,
                )
            except Exception:
                logger.exception("running transition failed for task %s", task_id)
                claimed = False
            if not claimed:
                logger.warning(
                    "Launch of task %s aborted: row no longer pending", task_id
                )
                with contextlib.suppress(Exception):
                    self._kill_process_group(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
                self._cleanup(task_id)
                return False

        reader = threading.Thread(
            target=self._read_pty, args=(task_id, master_fd), daemon=True
        )
        with self._lock:
            self._readers[task_id] = reader
        reader.start()
        return True

    def _read_pty(self, task_id: int, master_fd: int) -> None:
        path = TASK_LOGS_DIR / f"{task_id}.log"
        try:
            with open(path, "ab") as f:
                while True:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    try:
                        f.write(data)
                        f.flush()
                        self._line_cache.feed(path)
                    except OSError:
                        logger.warning("log write failed for task %s", task_id)
                        break
        except Exception:
            # Don't transition — _reap owns the terminal transition and records
            # the real exit code; transitioning here races it and loses the code.
            logger.exception("reader thread fatal error for task %s", task_id)
        # Close only if we still own the registration: _cleanup may already
        # have popped (and closed) the fd, and a second blind close could hit
        # an fd number since reused by another task's openpty.
        with self._lock:
            fd = self._master_fds.pop(task_id, None)
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _build_cli_args(
        self,
        model_name: str,
        params: dict,
    ) -> list[str]:
        """Build a CLI invocation from frontend form values.

        Dispatches on ``params["task_kind"]``: ``"optuna"`` launches a
        hyperparameter search via ``optuna_search.py`` (emitting
        ``--optuna_search.optuna_config``/``--optuna_search.metric``/
        ``--optuna_search.output_dir``), anything else runs ``train.py``. Both
        scripts share the same RunConfig flag contract, so model/data/general
        params are routed identically via the cached schema routes;
        default-equal params are omitted.
        """
        routes = self._schema_extractor.get_field_routes(model_name)
        defaults = self._schema_extractor.get_field_defaults(model_name)

        is_search = params.get("task_kind") == "optuna"
        script = "optuna_search.py" if is_search else "train.py"
        args = [script, "-m", model_name]

        if is_search:
            config_path = params.get("optuna_config_path")
            if config_path:
                args.extend(["--optuna_search.optuna_config", str(config_path)])
            metric = params.get("metric")
            if metric:
                args.extend(["--optuna_search.metric", str(metric)])
            output_dir = params.get("output_dir")
            if output_dir:
                args.extend(["--optuna_search.output_dir", str(output_dir)])

        dataset = params.get("dataset")
        if dataset:
            args.extend(["-d", str(dataset)])

        # build_param_flags skips fields without a RunConfig route, so optuna-only
        # keys (task_kind/metric/optuna_config_path/output_dir) are excluded.
        form_params = {k: v for k, v in params.items() if k != "dataset"}
        args.extend(build_param_flags(form_params, routes, defaults))
        return args

    def preview_command(self, model_name: str, params: dict) -> str:
        """Return the CLI invocation that would be executed for these params."""
        return " ".join(self._build_cli_args(model_name, params))

    def _kill_process_group(self, pid: int, sig: int) -> None:
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.kill(pid, sig)

    def _terminate(self, task_id: int) -> bool:
        with self._lock:
            proc = self._running.get(task_id)
        if proc is None:
            return False

        pid = proc.pid
        with contextlib.suppress(Exception):
            self._kill_process_group(pid, signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._kill_process_group(pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=3)

        if proc.poll() is None:
            # Stuck in uninterruptible sleep; leave it tracked so _reap keeps
            # polling instead of orphaning the Popen handle.
            logger.warning(
                "task %s did not die after SIGKILL; leaving tracked", task_id
            )
            return False
        self._cleanup(task_id)
        return True

    def _force_kill(self, task_id: int) -> None:
        # Wait for the child before dropping its handle: an unreaped Popen
        # leaves a zombie, and the row leaves _slot_usage as soon as it goes
        # to "stopped", so the scheduler would dispatch onto a GPU the dying
        # trainer still holds.
        with self._lock:
            proc = self._running.get(task_id)
        if proc is None:
            return

        with contextlib.suppress(Exception):
            self._kill_process_group(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            logger.warning(
                "task %s still alive after SIGKILL; freeing its slot anyway", task_id
            )
        self._cleanup(task_id)

    def _cleanup(self, task_id: int) -> None:
        # Join the reader before closing master_fd so the PTY kernel buffer
        # drains fully and the final log bytes are not lost.
        reader = self._readers.pop(task_id, None)
        if reader and reader.is_alive():
            reader.join(timeout=5)
        master_fd = self._master_fds.pop(task_id, None)
        if master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(master_fd)
        with self._lock:
            self._running.pop(task_id, None)

    def _reap(self) -> None:
        with self._lock:
            snapshot = list(self._running.items())
        for task_id, proc in snapshot:
            rc = proc.poll()
            if rc is None:
                continue
            # Drain the reader before the status flip: the live WS stream keys
            # check_alive off pid/status, so the final exit bytes must reach
            # the cache while the task still reads as running (mirrors
            # PreprocessManager._monitor).
            self._cleanup(task_id)
            to = "completed" if rc == 0 else "failed"
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "running",
                    to,
                    exit_code=rc,
                    finished_at=datetime.now(),
                    pid=None,
                )

    def stop_task(self, task_id: int) -> bool:
        """Gracefully stop a task (queue removal or SIGINT)."""
        if self.remove_from_queue(task_id):
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    finished_at=datetime.now(),
                )
            self._wake.set()
            return True

        with SessionLocal() as session:
            task = session.get(Task, task_id)
            status = task.status if task else None
            pid = task.pid if task else None
            started_at = task.started_at if task else None

        if status == "interrupted":
            if pid:
                self._kill_orphan_pid(pid, started_at)
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "interrupted",
                    "stopped",
                    finished_at=datetime.now(),
                    pid=None,
                )
            with self._lock:
                self._recover_monitors.pop(task_id, None)
            return True

        if status == "running":
            with SessionLocal() as session:
                claimed = transition(session, Task, task_id, "running", "stopping")
            if not claimed:
                return True
            self._terminate(task_id)
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "stopping",
                    "stopped",
                    finished_at=datetime.now(),
                    pid=None,
                )
            self._wake.set()
            return True

        if status == "pending":
            with SessionLocal() as session:
                return transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    finished_at=datetime.now(),
                )

        return False

    def kill_task(self, task_id: int) -> bool:
        """Force-kill a task via SIGKILL to its process group."""
        if self.remove_from_queue(task_id):
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    exit_code=-9,
                    finished_at=datetime.now(),
                )
            self._wake.set()
            return True

        with SessionLocal() as session:
            task = session.get(Task, task_id)
            status = task.status if task else None
            pid = task.pid if task else None
            started_at = task.started_at if task else None

        if status == "interrupted":
            if pid:
                self._kill_orphan_pid(pid, started_at)
            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "interrupted",
                    "stopped",
                    finished_at=datetime.now(),
                    pid=None,
                )
            with self._lock:
                self._recover_monitors.pop(task_id, None)
            return True

        if status == "running":
            with SessionLocal() as session:
                claimed = transition(session, Task, task_id, "running", "stopping")
            if not claimed:
                return True

            self._force_kill(task_id)

            with SessionLocal() as session:
                transition(
                    session,
                    Task,
                    task_id,
                    "stopping",
                    "stopped",
                    exit_code=-9,
                    finished_at=datetime.now(),
                    pid=None,
                )
            self._wake.set()
            return True

        if status == "pending":
            with SessionLocal() as session:
                return transition(
                    session,
                    Task,
                    task_id,
                    "pending",
                    "stopped",
                    exit_code=-9,
                    finished_at=datetime.now(),
                )

        return False

    def _kill_orphan_pid(self, pid: int, started_at: datetime | None) -> None:
        """SIGKILL a lingering pid's process group unless it was recycled.

        Interrupted rows can outlive their process (a stuck recover monitor
        keeps the pid around), so verify identity before killing what may now
        be an unrelated process that reused the pid.
        """
        try:
            proc = psutil.Process(pid)
            if pid_reused(proc, started_at):
                logger.warning("pid %s reused by another process; not killing", pid)
                return
        except psutil.Error:
            return
        # The orphan leader owns its session (start_new_session), so killing
        # the group also reaps worker subprocesses a bare kill would miss.
        self._kill_process_group(pid, signal.SIGKILL)

    def force_cleanup_interrupted(self, task_id: int) -> None:
        """SIGKILL a lingering interrupted task's pid and drop its recover monitor.

        Used by the delete endpoint to reclaim an interrupted task whose orphan
        process and recover-monitor thread would otherwise outlive its row.
        """
        with SessionLocal() as session:
            task = session.get(Task, task_id)
            pid = task.pid if task else None
            started_at = task.started_at if task else None
        if pid:
            self._kill_orphan_pid(pid, started_at)
        with self._lock:
            self._recover_monitors.pop(task_id, None)

    def recover_tasks(self) -> None:
        """Reattach live orphans, re-queue dead in-flight tasks, then queue pending.

        In-flight tasks whose process is gone (interrupted by a prior shutdown,
        or crashed) are put back to ``pending`` so they re-run instead of being
        lost; live orphans are re-attached. Pending tasks are then queued in
        ``id`` order, which matches creation (fold) order and keeps the queue
        stable across restarts.
        """
        with SessionLocal() as session:
            inflight = (
                session.query(Task.id, Task.pid, Task.status, Task.started_at)
                .filter(Task.status.in_(["running", "stopping", "interrupted"]))
                .all()
            )
            for task_id, pid, prior_status, started_at in inflight:
                if pid and psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running() and not pid_reused(proc, started_at):
                            if prior_status != "interrupted":
                                transition(
                                    session, Task, task_id, prior_status, "interrupted"
                                )
                            t = threading.Thread(
                                target=self._recover_monitor,
                                args=(task_id, pid),
                                daemon=True,
                            )
                            t.start()
                            with self._lock:
                                self._recover_monitors[task_id] = t
                            continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                        pass
                transition(
                    session,
                    Task,
                    task_id,
                    prior_status,
                    "pending",
                    pid=None,
                    gpu_assigned=None,
                    started_at=None,
                    finished_at=None,
                    exit_code=None,
                )

            pending = (
                session.query(Task.id, Task.gpu_request)
                .filter(Task.status == "pending")
                .order_by(Task.id)
                .all()
            )
            with self._lock:
                for task_id, gpu_request in pending:
                    self._queue.append((task_id, gpu_request))
        self._wake.set()

    def _recover_monitor(self, task_id: int, pid: int) -> None:
        exit_code = -1
        try:
            proc = psutil.Process(pid)
            # psutil.Process has no returncode attribute; wait() returns the
            # exit status, or None for non-child pids (recovered orphans are
            # never children). None keeps the unknown outcome honest; the
            # status below still resolves conservatively to failed.
            exit_code = proc.wait()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            OSError,
        ):
            pass

        with SessionLocal() as session:
            task = session.get(Task, task_id)
            if task and task.status in ("running", "interrupted"):
                to = "completed" if exit_code == 0 else "failed"
                transition(
                    session,
                    Task,
                    task_id,
                    task.status,
                    to,
                    exit_code=exit_code,
                    finished_at=datetime.now(),
                    pid=None,
                )
        with self._lock:
            self._recover_monitors.pop(task_id, None)

    def shutdown(self) -> None:
        """Stop the scheduler, gracefully stop running tasks, mark them interrupted.

        Each running task gets the same graceful stop the UI uses (SIGINT to its
        process group, then SIGKILL only if it does not exit in time), so the
        trainer can clean up and no orphans survive the backend exit. ``pid``
        stays on the row: on the next start ``recover_tasks`` re-attaches live
        orphans by pid and re-queues only rows whose process is gone.
        """
        self._stopping = True
        self._wake.set()
        # The scheduler may be mid-launch (cold schema extraction runs a
        # subprocess for up to 60s). Loop-terminate until the thread has
        # exited so no Popen can be registered after the sweep — a launch
        # completing during shutdown would otherwise survive it and be
        # double-dispatched on the next start. Bounded so a wedged launch
        # cannot hang server shutdown.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                self._thread.join(timeout=0.5)
            with self._lock:
                running = list(self._running)
            for task_id in running:
                self._terminate(task_id)
            with self._lock:
                drained = not self._running
            if not self._thread.is_alive() and drained:
                break

        with SessionLocal() as session:
            # Keep pid: recover_tasks re-attaches survivors instead of
            # re-queueing them alongside a still-running orphan.
            session.query(Task).filter(Task.status.in_(["running", "stopping"])).update(
                {"status": "interrupted"}
            )
            session.commit()
