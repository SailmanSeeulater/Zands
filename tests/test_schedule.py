import plistlib
import subprocess
from pathlib import Path

import pytest

from zands import schedule


class FakeLaunchctl:
    """Records launchctl calls; `loaded` controls what `print` reports."""

    def __init__(self, loaded=False, bootstrap_rc=0):
        self.calls = []
        self.loaded = loaded
        self.bootstrap_rc = bootstrap_rc

    def __call__(self, args, **kwargs):
        self.calls.append(args[1:])
        verb = args[1]
        if verb == "print":
            out = "\tstate = not running\n\truns = 3\n\tlast exit code = 0\n" if self.loaded else ""
            return subprocess.CompletedProcess(args, 0 if self.loaded else 113, out, "")
        if verb == "bootstrap":
            self.loaded = self.bootstrap_rc == 0
            return subprocess.CompletedProcess(args, self.bootstrap_rc, "", "Bootstrap failed: 5: Input/output error")
        if verb == "bootout":
            self.loaded = False
        return subprocess.CompletedProcess(args, 0, "", "")


def test_build_plist(tmp_path):
    p = schedule.build_plist("data/news.db", "feeds.toml", interval_minutes=60, python="/venv/bin/python",
                             workdir=tmp_path, log=tmp_path / "ingest.log", env={"OLLAMA_HOST": "http://h:1", "HOME": "/x"})
    assert p["Label"] == schedule.LABEL
    args = p["ProgramArguments"]
    assert args[:3] == ["/venv/bin/python", "-m", "zands.cli"]
    assert args[3] == "--db" and Path(args[4]).is_absolute() and args[4].endswith("data/news.db")
    assert args[5:7] == ["ingest", "--feeds"] and Path(args[7]).is_absolute()
    assert p["StartInterval"] == 3600 and p["RunAtLoad"] is True
    assert p["StandardOutPath"] == p["StandardErrorPath"] == str(tmp_path / "ingest.log")
    assert p["EnvironmentVariables"]["OLLAMA_HOST"] == "http://h:1"
    assert "HOME" not in p["EnvironmentVariables"]  # only known Zands/Ollama variables pass through
    with pytest.raises(ValueError):
        schedule.build_plist("d", "f", interval_minutes=1)


def test_write_and_status(tmp_path):
    plist = schedule.build_plist("d.db", "f.toml", log=tmp_path / "logs" / "ingest.log", env={})
    path = schedule.write_plist(plist, tmp_path / "agent.plist")
    assert plistlib.loads(path.read_bytes())["Label"] == schedule.LABEL
    assert (tmp_path / "logs").is_dir()
    (tmp_path / "logs" / "ingest.log").write_text("line1\nline2\n")
    st = schedule.status(runner=FakeLaunchctl(loaded=True), path=path)
    assert st.installed and st.loaded and st.state == "not running" and st.runs == 3 and st.last_exit == "0"
    assert st.interval_minutes == 60 and st.log_tail == ["line1", "line2"]
    st = schedule.status(runner=FakeLaunchctl(loaded=False), path=tmp_path / "missing.plist")
    assert not st.installed and not st.loaded


def test_load_reloads_and_reports_errors(tmp_path):
    fake = FakeLaunchctl(loaded=True)
    schedule.load(tmp_path / "agent.plist", runner=fake)
    verbs = [c[0] for c in fake.calls]
    assert verbs == ["print", "bootout", "bootstrap"]
    assert fake.calls[2] == ["bootstrap", schedule.domain(), str(tmp_path / "agent.plist")]
    with pytest.raises(RuntimeError, match="Input/output"):
        schedule.load(tmp_path / "agent.plist", runner=FakeLaunchctl(bootstrap_rc=5))


def test_uninstall(tmp_path):
    path = tmp_path / "agent.plist"
    path.write_bytes(b"x")
    fake = FakeLaunchctl(loaded=True)
    assert schedule.uninstall(runner=fake, path=path)
    assert not path.exists() and ["bootout", f"{schedule.domain()}/{schedule.LABEL}"] in fake.calls
    assert not schedule.uninstall(runner=FakeLaunchctl(), path=path)
