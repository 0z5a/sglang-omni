#!/usr/bin/env python3
"""Average nsys GPU metrics over the timed SeedTTS cohort only.

The headline bench.log must contain:
  Warmup (16 requests)...
  Benchmarking 200 requests ...
  Results saved to ...

The SM window is [Benchmarking, Results saved]. Probe, preload, and warmup
are excluded.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path

METRIC_IDS = {
    2: "GR Active",
    3: "SMs Active",
    4: "SM Issue",
    5: "Tensor Active",
}

BENCH_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)\s+"
    r".*Benchmarking (\d+) requests"
)
SAVED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)\s+.*Results saved to"
)


def _parse_log_ts(date_s: str, ms: str) -> datetime:
    return datetime.fromisoformat(f"{date_s.replace(' ', 'T')}.{ms}")


def parse_headline_window(bench_log: Path) -> tuple[datetime, datetime, int]:
    start: datetime | None = None
    end: datetime | None = None
    n_requests = 0
    for line in bench_log.read_text().splitlines():
        bench = BENCH_RE.match(line)
        if bench:
            start = _parse_log_ts(bench.group(1), bench.group(2))
            n_requests = int(bench.group(3))
            continue
        saved = SAVED_RE.match(line)
        if saved and start is not None:
            end = _parse_log_ts(saved.group(1), saved.group(2))
            break
    if start is None or end is None:
        raise ValueError(f"could not find Benchmarking/Results saved in {bench_log}")
    return start, end, n_requests


def session_start(sqlite_path: Path) -> datetime:
    con = sqlite3.connect(sqlite_path)
    utc = con.execute("SELECT utcTime FROM TARGET_INFO_SESSION_START_TIME").fetchone()
    con.close()
    if utc is None:
        raise ValueError(f"no TARGET_INFO_SESSION_START_TIME in {sqlite_path}")
    return datetime.fromisoformat(utc[0])


def metric_mean(
    sqlite_path: Path, metric_id: int, t0_ns: int, t1_ns: int
) -> tuple[float, int]:
    con = sqlite3.connect(sqlite_path)
    rows = [
        v
        for (v,) in con.execute(
            "SELECT value FROM GPU_METRICS "
            "WHERE metricId=? AND timestamp>=? AND timestamp<=?",
            (metric_id, t0_ns, t1_ns),
        )
    ]
    con.close()
    if not rows:
        raise ValueError(f"no GPU_METRICS samples for metricId={metric_id}")
    return sum(rows) / len(rows), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--bench-log", type=Path, required=True)
    args = parser.parse_args()

    bench0, bench1, n_requests = parse_headline_window(args.bench_log)
    sess = session_start(args.sqlite)
    t0_ns = int((bench0 - sess).total_seconds() * 1e9)
    t1_ns = int((bench1 - sess).total_seconds() * 1e9)
    if t1_ns <= t0_ns:
        raise ValueError(
            f"empty window: session={sess} bench=[{bench0}, {bench1}]"
        )

    print(f"session_start     {sess.isoformat()}")
    print(f"bench_start       {bench0.isoformat()}  (after warmup)")
    print(f"bench_end         {bench1.isoformat()}")
    print(f"timed_requests    {n_requests}")
    print(f"session_rel_s     {t0_ns / 1e9:.3f} .. {t1_ns / 1e9:.3f}")
    print(f"window_s          {(t1_ns - t0_ns) / 1e9:.3f}")
    for metric_id, name in METRIC_IDS.items():
        mean, n = metric_mean(args.sqlite, metric_id, t0_ns, t1_ns)
        print(f"{name:16s}  {mean:.4f}%  n={n}")


if __name__ == "__main__":
    main()
