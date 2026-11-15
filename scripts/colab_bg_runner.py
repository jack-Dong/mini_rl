"""Helper run inside Colab via `colab exec -f` to launch or monitor detached jobs."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def launch_job(cmd: str, log_file: Path, done_file: Path, pid_file: Path) -> None:
    """Start a double-forked nohup command writing logs, PID, and exit code markers."""
    for path in (log_file, done_file, pid_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    inner = f"{cmd}; echo $? > {shlex.quote(str(done_file))}"
    outer = (
        f"nohup bash -lc {shlex.quote(inner)} "
        f"> {shlex.quote(str(log_file))} 2>&1 < /dev/null & "
        f"echo $! > {shlex.quote(str(pid_file))}"
    )
    subprocess.run(["bash", "-c", outer], check=True)
    pid = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else "none"
    print(f"LAUNCHED pid={pid} log={log_file} done={done_file}")


def _is_alive(pid: str) -> bool:
    if not pid.isdigit():
        return False
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return False
    try:
        text = status_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        if line.startswith("State:") and "Z (zombie)" in line:
            return False
    return True


def check_status(log_file: Path, done_file: Path, pid_file: Path) -> None:
    """Print detached job state, completion exit code, and trailing log lines."""
    pid = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else ""
    done_code = done_file.read_text(encoding="utf-8").strip() if done_file.exists() else None
    alive = _is_alive(pid)

    print(f"STATUS pid={pid or 'none'} alive={alive} done_code={done_code}")
    if log_file.exists():
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        print("--- LOG TAIL ---")
        for line in lines[-25:]:
            print(line)


def main() -> None:
    """Dispatch `launch` or `status` using environment variables."""
    action = os.environ.get("ACTION", "status")
    log_file = Path(os.environ.get("LOG_FILE", "/content/train.log"))
    done_file = Path(os.environ.get("DONE_FILE", "/content/DONE"))
    pid_file = Path(os.environ.get("PID_FILE", "/content/train.pid"))

    if action == "launch":
        cmd = os.environ.get("CMD", "")
        if not cmd:
            raise ValueError("CMD environment variable must be non-empty for ACTION=launch")
        launch_job(cmd, log_file, done_file, pid_file)
    elif action == "status":
        check_status(log_file, done_file, pid_file)
    else:
        raise ValueError(f"Unsupported ACTION={action!r}")


if __name__ == "__main__":
    main()
