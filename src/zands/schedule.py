"""Hourly ingest via a launchd user agent (macOS).

launchd rather than cron: cron silently skips runs while the laptop sleeps;
launchd runs a missed StartInterval job once on wake.

The agent runs this venv's Python directly (`python -m zands.cli ... ingest`),
so it needs neither uv nor a shell, and uses absolute paths for the database
and feeds file.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LABEL = "local.zands.ingest"
DEFAULT_INTERVAL_MINUTES = 60
PASSTHROUGH_ENV = ("OLLAMA_HOST", "ZANDS_MODEL", "ZANDS_EMBED_MODEL")

Runner = Callable[..., subprocess.CompletedProcess]


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "zands" / "ingest.log"


def domain() -> str:
    return f"gui/{os.getuid()}"


def build_plist(
    db: str | Path,
    feeds: str | Path,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    python: str | None = None,
    workdir: str | Path | None = None,
    log: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    if interval_minutes < 5:
        raise ValueError("interval must be at least 5 minutes")
    log = str(log or log_path())
    environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin", "PYTHONUNBUFFERED": "1"}
    source_env = os.environ if env is None else env
    environment.update({k: source_env[k] for k in PASSTHROUGH_ENV if k in source_env})
    return {
        "Label": LABEL,
        "ProgramArguments": [
            python or sys.executable, "-m", "zands.cli",
            "--db", str(Path(db).resolve()),
            "ingest", "--feeds", str(Path(feeds).resolve()),
        ],
        "WorkingDirectory": str(Path(workdir or Path.cwd()).resolve()),
        "StartInterval": interval_minutes * 60,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Nice": 10,
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": environment,
    }


def write_plist(plist: dict, path: Path | None = None) -> Path:
    path = path or plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(plist["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        plistlib.dump(plist, fh)
    return path


def _run(runner: Runner, *args: str) -> subprocess.CompletedProcess:
    return runner(["launchctl", *args], capture_output=True, text=True)


def is_loaded(runner: Runner = subprocess.run) -> bool:
    return _run(runner, "print", f"{domain()}/{LABEL}").returncode == 0


def load(path: Path | None = None, runner: Runner = subprocess.run) -> None:
    """(Re)load the agent into launchd. Starts a first ingest right away (RunAtLoad)."""
    if is_loaded(runner):
        _run(runner, "bootout", f"{domain()}/{LABEL}")
    res = _run(runner, "bootstrap", domain(), str(path or plist_path()))
    if res.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {(res.stderr or res.stdout).strip()}")


def uninstall(runner: Runner = subprocess.run, path: Path | None = None) -> bool:
    """Unload the agent and delete its plist. Returns True if anything was removed."""
    path = path or plist_path()
    removed = False
    if is_loaded(runner):
        _run(runner, "bootout", f"{domain()}/{LABEL}")
        removed = True
    if path.exists():
        path.unlink()
        removed = True
    return removed


@dataclass
class Status:
    installed: bool
    loaded: bool
    state: str | None = None
    runs: int | None = None
    last_exit: str | None = None
    interval_minutes: int | None = None
    log: str | None = None
    log_tail: list[str] | None = None


def _field(text: str, name: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(name)} = (.+)$", text, re.M)
    return m.group(1).strip() if m else None


def status(runner: Runner = subprocess.run, path: Path | None = None, tail: int = 8) -> Status:
    path = path or plist_path()
    st = Status(installed=path.exists(), loaded=False)
    if st.installed:
        with open(path, "rb") as fh:
            plist = plistlib.load(fh)
        st.interval_minutes = plist.get("StartInterval", 0) // 60
        st.log = plist.get("StandardOutPath")
        if st.log and Path(st.log).exists():
            st.log_tail = Path(st.log).read_text(errors="replace").splitlines()[-tail:]
    res = _run(runner, "print", f"{domain()}/{LABEL}")
    if res.returncode == 0:
        st.loaded = True
        st.state = _field(res.stdout, "state")
        runs = _field(res.stdout, "runs")
        st.runs = int(runs) if runs and runs.isdigit() else None
        st.last_exit = _field(res.stdout, "last exit code")
    return st
