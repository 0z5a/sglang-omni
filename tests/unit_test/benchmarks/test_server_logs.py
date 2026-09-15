# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from benchmarks.benchmarker import utils


def test_tee_keeps_log_open_after_launch_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ready_path = tmp_path / "ready"
    log_path = tmp_path / "server.log"
    threads: list[threading.Thread] = []
    thread_class = threading.Thread

    def make_thread(**kwargs: object) -> threading.Thread:
        thread = thread_class(**kwargs)
        threads.append(thread)
        return thread

    def skip_health(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(utils.threading, "Thread", make_thread)
    monkeypatch.setattr(utils, "wait_healthy", skip_health)
    command = [
        sys.executable,
        "-c",
        "import pathlib, sys, time\n"
        "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
        "print('after launch', flush=True)",
        str(ready_path),
    ]
    process = utils.start_server_from_cmd(command, log_path, 0, tee=True)
    try:
        ready_path.touch()
        assert process.wait(timeout=10) == 0
        threads[0].join(timeout=10)
        assert not threads[0].is_alive()
        assert log_path.read_text() == "after launch\n"
        assert "after launch\n" in capsys.readouterr().out
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        threads[0].join(timeout=10)
