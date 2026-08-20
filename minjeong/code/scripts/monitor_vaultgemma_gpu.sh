#!/usr/bin/env bash
set -euo pipefail

exec /usr/bin/python3 - "$@" <<'PY'
import datetime
import errno
import math
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path


HEADER = (
    "timestamp_utc,owned_pid,gpu_index,gpu_uuid,memory_used_mib,"
    "utilization_gpu_percent,power_draw_watts,temperature_c,"
    "cpu_process_tree_rss_bytes\n"
)
SAFE_TEST_ROOT = re.compile(r"/tmp/vaultgemma-task9-monitor\.[A-Za-z0-9_-]+")


class ContractError(Exception):
    pass


class ProcessProbeError(Exception):
    pass


class PublicationError(Exception):
    pass


class OwnedProcessRootMissing(Exception):
    pass


def usage(message):
    print(f"Error: {message}", file=sys.stderr)
    print(
        f"Usage: {sys.argv[0]} <owned_pid> <gpu_index> <output_csv> "
        "[interval_seconds] [max_samples]",
        file=sys.stderr,
    )
    raise SystemExit(2)


def parse_arguments(arguments):
    if not 3 <= len(arguments) <= 5:
        usage("expected three to five arguments")
    pid_raw, gpu_raw, output_raw = arguments[:3]
    interval_raw = arguments[3] if len(arguments) >= 4 else "1"
    samples_raw = arguments[4] if len(arguments) >= 5 else "36000"
    if re.fullmatch(r"[1-9][0-9]*", pid_raw) is None:
        usage("owned_pid must be a positive integer")
    if re.fullmatch(r"0|[1-9][0-9]*", gpu_raw) is None:
        usage("gpu_index must be canonical")
    if re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", interval_raw) is None:
        usage("interval_seconds must be numeric")
    interval = float(interval_raw)
    if not math.isfinite(interval) or not 0.05 <= interval <= 60:
        usage("interval_seconds must be in 0.05..60")
    if re.fullmatch(r"[1-9][0-9]*", samples_raw) is None:
        usage("max_samples must be positive")
    if len(samples_raw) > 5 or int(samples_raw) > 86400:
        usage("max_samples must be in 1..86400")
    output = Path(output_raw)
    if not output.is_absolute() or output.name in {"", ".", ".."}:
        usage("output_csv must be an absolute file path")
    return int(pid_raw), int(gpu_raw), output, interval, int(samples_raw)


def validate_private_test_root(raw_root):
    if SAFE_TEST_ROOT.fullmatch(raw_root or "") is None:
        raise ContractError("test root is outside the monitor contract namespace")
    root = Path(raw_root)
    try:
        metadata = root.lstat()
    except OSError as error:
        raise ContractError("test root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ContractError("test root must be an owned directory")
    if stat.S_IMODE(metadata.st_mode) != 0o700 or str(root.resolve()) != str(root):
        raise ContractError("test root must be private and canonical")
    sentinel = root / ".vaultgemma-monitor-contract-test-owned"
    try:
        sentinel_metadata = sentinel.lstat()
    except OSError as error:
        raise ContractError("test ownership sentinel is unavailable") from error
    if (
        not stat.S_ISREG(sentinel_metadata.st_mode)
        or sentinel_metadata.st_uid != os.getuid()
        or stat.S_IMODE(sentinel_metadata.st_mode) != 0o600
        or sentinel_metadata.st_nlink != 1
    ):
        raise ContractError("test ownership sentinel is unsafe")
    return root


def validate_test_path(root, raw_path, *, executable, directory):
    path = Path(raw_path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError("test override path is unavailable") from error
    if str(path) != str(resolved) or root not in resolved.parents:
        raise ContractError("test override escaped its private root")
    metadata = path.lstat()
    if metadata.st_uid != os.getuid() or stat.S_ISLNK(metadata.st_mode):
        raise ContractError("test override is not owned or is a symlink")
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise ContractError("test proc root must be a directory")
    if not directory and not stat.S_ISREG(metadata.st_mode):
        raise ContractError("test hook must be a regular file")
    if executable and not os.access(path, os.X_OK):
        raise ContractError("test hook must be executable")
    return path


def test_configuration():
    names = (
        "VAULTGEMMA_MONITOR_TEST_MODE",
        "VAULTGEMMA_MONITOR_TEST_ROOT",
        "VAULTGEMMA_MONITOR_PROC_ROOT",
        "VAULTGEMMA_MONITOR_TEST_PROBE_HOOK",
        "VAULTGEMMA_MONITOR_TEST_BEFORE_CREATE",
        "VAULTGEMMA_MONITOR_TEST_BEFORE_PUBLISH",
    )
    configured = {name: os.environ.get(name) for name in names}
    if not any(value is not None for value in configured.values()):
        return Path("/proc"), None, None, None
    if configured["VAULTGEMMA_MONITOR_TEST_MODE"] != "contract":
        raise ContractError("monitor test overrides require explicit contract mode")
    root = validate_private_test_root(configured["VAULTGEMMA_MONITOR_TEST_ROOT"])
    proc_root_raw = configured["VAULTGEMMA_MONITOR_PROC_ROOT"]
    proc_root = (
        validate_test_path(root, proc_root_raw, executable=False, directory=True)
        if proc_root_raw
        else Path("/proc")
    )
    probe_hook_raw = configured["VAULTGEMMA_MONITOR_TEST_PROBE_HOOK"]
    create_hook_raw = configured["VAULTGEMMA_MONITOR_TEST_BEFORE_CREATE"]
    publish_hook_raw = configured["VAULTGEMMA_MONITOR_TEST_BEFORE_PUBLISH"]
    probe_hook = (
        validate_test_path(root, probe_hook_raw, executable=True, directory=False)
        if probe_hook_raw
        else None
    )
    create_hook = (
        validate_test_path(root, create_hook_raw, executable=True, directory=False)
        if create_hook_raw
        else None
    )
    publish_hook = (
        validate_test_path(root, publish_hook_raw, executable=True, directory=False)
        if publish_hook_raw
        else None
    )
    return proc_root, probe_hook, create_hook, publish_hook


def run_hook(hook, *arguments):
    if hook is None:
        return
    try:
        subprocess.run([str(hook), *map(str, arguments)], check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError("monitor test hook failed") from error


class OwnedProcessProbe:
    ALIVE = "alive"
    GONE = "gone_or_reused"
    ERROR = "error"

    def __init__(self, proc_root, pid, hook):
        self.proc_root = proc_root
        self.pid = pid
        self.hook = hook
        self.expected_uid = os.getuid()
        self.probe_count = 0

    @property
    def pid_directory(self):
        return self.proc_root / str(self.pid)

    @property
    def stat_path(self):
        return self.pid_directory / "stat"

    def _entry_still_exists(self):
        return os.path.lexists(self.pid_directory)

    def _gone_or_error(self, context, error):
        if not self._entry_still_exists():
            return self.GONE, None, None
        return self.ERROR, f"{context}: {type(error).__name__}", None

    def inspect(self, expected_start_time=None):
        self.probe_count += 1
        run_hook(
            self.hook, "before_pid_stat", self.probe_count,
            self.pid_directory, self.stat_path,
        )
        try:
            pid_metadata = os.stat(self.pid_directory)
        except OSError as error:
            return self._gone_or_error("cannot stat owned PID directory", error)
        if not stat.S_ISDIR(pid_metadata.st_mode):
            return self.ERROR, "owned PID entry is not a directory", None
        run_hook(
            self.hook, "after_pid_stat", self.probe_count,
            self.pid_directory, self.stat_path,
        )
        try:
            stat_bytes = self.stat_path.read_bytes()
        except OSError as error:
            return self._gone_or_error("cannot read owned PID stat", error)
        try:
            prefix = f"{self.pid} (".encode("ascii")
            separator = stat_bytes.rfind(b") ")
            if not stat_bytes.startswith(prefix) or separator < len(prefix):
                raise ValueError("PID/stat prefix mismatch")
            fields = stat_bytes[separator + 2 :].split()
            if len(fields) < 20 or not fields[19].isdigit():
                raise ValueError("missing numeric start time")
            start_time = fields[19].decode("ascii")
        except (UnicodeError, ValueError) as error:
            return self._gone_or_error("malformed owned PID stat", error)
        if expected_start_time is not None and start_time != expected_start_time:
            return self.GONE, None, start_time
        if pid_metadata.st_uid != self.expected_uid:
            return self.ERROR, "owned PID UID changed or is not current-user owned", start_time
        return self.ALIVE, None, start_time


def open_directory_chain(path):
    parts = path.parts
    if not path.is_absolute() or not parts or parts[0] != "/":
        raise PublicationError("CSV parent must be absolute")
    if any(part in {"", ".", ".."} for part in parts[1:]):
        raise PublicationError("CSV parent path is not canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise PublicationError("CSV parent descriptor is not a directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def write_all(descriptor, payload):
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PublicationError("CSV write made no progress")
        view = view[written:]


def open_csv_no_replace(output, create_hook):
    parent_descriptor = open_directory_chain(output.parent)
    file_descriptor = -1
    try:
        run_hook(create_hook, "before_create", output.parent, output.name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(
                output.name, flags, 0o600, dir_fd=parent_descriptor
            )
        except FileExistsError as error:
            raise PublicationError("refusing to overwrite an existing CSV target") from error
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PublicationError("created CSV target is not one regular file")
        write_all(file_descriptor, HEADER.encode("ascii"))
        os.fsync(file_descriptor)
        os.fsync(parent_descriptor)
        return parent_descriptor, file_descriptor
    except Exception:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(parent_descriptor)
        raise


def process_tree_rss_bytes(root_pid):
    try:
        completed = subprocess.run(
            ["ps", "-e", "-o", "pid=,ppid=,rss="],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("CPU RSS sampling failed") from error
    parent = {}
    rss = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise RuntimeError("CPU RSS process table is malformed")
        pid, ppid, rss_kib = map(int, fields)
        parent[pid] = ppid
        rss[pid] = rss_kib
    if root_pid not in parent:
        raise OwnedProcessRootMissing("owned process root missing from process table")
    included = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if pid not in included and ppid in included:
                included.add(pid)
                changed = True
    return sum(rss[pid] for pid in included) * 1024


def sample_gpu(gpu_index):
    command = [
        "nvidia-smi", f"--id={gpu_index}",
        "--query-gpu=uuid,memory.used,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"nvidia-smi sampling failed: {type(error).__name__}") from error
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi sampling failed with exit {completed.returncode}")
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi did not return exactly one GPU row")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5 or re.fullmatch(r"GPU-[A-Za-z0-9_-]+", fields[0]) is None:
        raise RuntimeError("nvidia-smi returned a malformed GPU identity")
    for value in fields[1:]:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None:
            raise RuntimeError("nvidia-smi returned a malformed numeric field")
    return fields


def main():
    owned_pid, gpu_index, output, interval, max_samples = parse_arguments(sys.argv[1:])
    try:
        proc_root, probe_hook, create_hook, publish_hook = test_configuration()
    except ContractError as error:
        usage(str(error))
    probe = OwnedProcessProbe(proc_root, owned_pid, probe_hook)
    state, detail, initial_start_time = probe.inspect()
    if state != probe.ALIVE:
        usage(detail or "owned_pid is absent, reused, or not current-user owned")
    parent_descriptor = -1
    csv_descriptor = -1
    try:
        parent_descriptor, csv_descriptor = open_csv_no_replace(output, create_hook)
        sample_count = 0
        while True:
            state, detail, _ = probe.inspect(initial_start_time)
            if state == probe.GONE:
                break
            if state == probe.ERROR:
                raise ProcessProbeError(f"owned process probe error: {detail}")
            if sample_count >= max_samples:
                raise RuntimeError(
                    "maximum sample count reached while owned process is still alive"
                )
            gpu_uuid, memory, utilization, power, temperature = sample_gpu(gpu_index)
            try:
                cpu_rss = process_tree_rss_bytes(owned_pid)
            except OwnedProcessRootMissing as error:
                state, detail, _ = probe.inspect(initial_start_time)
                if state == probe.GONE:
                    break
                if state == probe.ERROR:
                    raise ProcessProbeError(f"owned process probe error: {detail}") from error
                raise RuntimeError(str(error)) from error
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            row = (
                f"{timestamp},{owned_pid},{gpu_index},{gpu_uuid},{memory},"
                f"{utilization},{power},{temperature},{cpu_rss}\n"
            ).encode("ascii")
            run_hook(
                publish_hook, "before_publish", sample_count + 1,
                probe.pid_directory, probe.stat_path,
            )
            state, detail, _ = probe.inspect(initial_start_time)
            if state == probe.GONE:
                break
            if state == probe.ERROR:
                raise ProcessProbeError(f"owned process probe error: {detail}")
            write_all(csv_descriptor, row)
            os.fsync(csv_descriptor)
            sample_count += 1
            time.sleep(interval)
        os.fsync(csv_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if csv_descriptor >= 0:
            os.close(csv_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


try:
    main()
except PublicationError as error:
    print(f"Error: CSV publication failed: {error}", file=sys.stderr)
    raise SystemExit(3)
except ProcessProbeError as error:
    print(f"Error: {error}", file=sys.stderr)
    raise SystemExit(7)
except (ContractError, RuntimeError) as error:
    print(f"Error: {error}", file=sys.stderr)
    raise SystemExit(5)
PY
