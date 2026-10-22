#!/usr/bin/env python3
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/viztracer/blob/master/NOTICE.txt

"""
Run the test suite against a manylinux aarch64 wheel in python:*-slim
containers with CI-like resources, using the same test image the release
gate builds. Every run uses a fresh container with --network none.

Five consecutive plain runs, then five coverage runs, on one CPU and 2 GB:

    python tools/container_run.py --wheel dist/viztracer-...-cp314-cp314-manylinux...whl \\
        --output-dir out/ --cpus 1 --memory 2g --repeat 5
    python tools/container_run.py --wheel ... --output-dir out/ --cpus 1 --memory 2g --repeat 5 --coverage

One run each with and without --cap-add SYS_PTRACE and --init, failing unless
every test has the same outcome in all four:

    python tools/container_run.py --wheel ... --output-dir out/ --variants

The docker commands are printed and saved in OUTPUT_DIR/container-runs.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_gate  # noqa: E402

VARIANTS = {
    "default": [],
    "sys-ptrace": ["--cap-add", "SYS_PTRACE"],
    "init": ["--init"],
    "sys-ptrace-init": ["--cap-add", "SYS_PTRACE", "--init"],
}


def build_image(wheel: Path, image: str, workdir: Path) -> None:
    tag = wheel.name.split("-")[3]
    if (
        tag not in release_gate.LINUX_TEST_IMAGES
        or "manylinux" not in wheel.name
        or "aarch64" not in wheel.name
    ):
        raise SystemExit(
            f"{wheel.name} is not a {'/'.join(release_gate.LINUX_TEST_IMAGES)} manylinux aarch64 wheel"
        )
    base, uv_python = release_gate.LINUX_TEST_IMAGES[tag]
    downloads = workdir / "downloads"
    downloads.mkdir()
    context = release_gate.build_image_context(
        workdir / "context", release_gate.SOURCE_ROOT, wheel, downloads, base, uv_python
    )
    print(f"$ docker build -t {image} {context}", flush=True)
    subprocess.run(
        ["docker", "build", "--network", "default", "-q", "-t", image, str(context)],
        check=True,
    )


def docker_run(
    image: str, results: Path, label: str, flags: list, args: argparse.Namespace
) -> dict:
    results.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--network", "none", *flags]
    if args.cpus:
        cmd += [f"--cpus={args.cpus}"]
    if args.memory:
        cmd += [f"--memory={args.memory}"]
    cmd += ["-v", f"{results}:/results", image]
    cmd += [
        "python",
        "/opt/vt/src/tools/test_matrix.py",
        "run",
        "--root",
        "/opt/vt/root",
    ]
    cmd += ["--results-dir", "/results", "--label", label, "--envs", *args.envs]
    cmd += ["--chown", f"{os.getuid()}:{os.getgid()}"]
    if args.coverage:
        cmd.append("--coverage")
    print("$ " + " ".join(cmd), flush=True)
    start = time.monotonic()
    proc = subprocess.run(cmd)
    elapsed = time.monotonic() - start
    summary_path = results / f"summary-{label}.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else {"ok": False, "runs": []}
    )
    outcomes = {}
    for run in summary["runs"]:
        outcome_file = results / Path(run["outcomes"]).name
        if outcome_file.exists():
            for test, outcome in json.loads(outcome_file.read_text()).items():
                outcomes[f"{run['environment']}::{test}"] = outcome
    record = {
        "label": label,
        "command": cmd,
        "returncode": proc.returncode,
        "wall_seconds": round(elapsed, 1),
        "ok": proc.returncode == 0
        and summary["ok"]
        and elapsed < args.max_minutes * 60,
        "runs": [
            {
                k: r.get(k)
                for k in (
                    "id",
                    "passed",
                    "failures",
                    "errors",
                    "skipped",
                    "duration_seconds",
                    "ok",
                )
            }
            for r in summary["runs"]
        ],
    }
    status = "ok" if record["ok"] else "FAILED"
    print(
        f"[container_run] {label}: {status} in {record['wall_seconds']}s {record['runs']}",
        flush=True,
    )
    return {**record, "_outcomes": outcomes}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--wheel",
        required=True,
        help="manylinux aarch64 wheel (cp312, cp314 or cp314t)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="where logs, JUnit XML and container-runs.json go",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="consecutive runs, each in a fresh container",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="run the way CI measures coverage"
    )
    parser.add_argument("--cpus", help="docker --cpus")
    parser.add_argument("--memory", help="docker --memory")
    parser.add_argument(
        "--cap-add", action="append", default=[], help="docker --cap-add"
    )
    parser.add_argument("--init", action="store_true", help="docker --init")
    parser.add_argument(
        "--variants",
        action="store_true",
        help="compare runs with and without SYS_PTRACE and --init",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["full"],
        help="prepared environments to run (default: full)",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=15,
        help="fail a run that takes longer (default 15)",
    )
    parser.add_argument(
        "--keep-image", action="store_true", help="do not remove the test image"
    )
    args = parser.parse_args(argv)

    wheel = Path(args.wheel).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = f"viztracer-container-run-{uuid.uuid4().hex[:8]}"
    workdir = Path(tempfile.mkdtemp(prefix="viztracer-container-run-"))
    records = []
    ok = True
    try:
        build_image(wheel, image, workdir)
        mode = "coverage" if args.coverage else "plain"
        if args.variants:
            for name, flags in VARIANTS.items():
                records.append(
                    docker_run(
                        image, output / f"{mode}-{name}", f"{mode}-{name}", flags, args
                    )
                )
            reference = records[0]["_outcomes"]
            for record in records[1:]:
                diff = sorted(
                    t
                    for t in set(reference) | set(record["_outcomes"])
                    if reference.get(t) != record["_outcomes"].get(t)
                )
                record["identical_to_default"] = not diff
                if diff:
                    ok = False
                    print(
                        f"[container_run] {record['label']} differs from default: {diff[:20]}"
                    )
            ok = ok and bool(reference)
        else:
            flags = [f for cap in args.cap_add for f in ("--cap-add", cap)]
            if args.init:
                flags.append("--init")
            for i in range(1, args.repeat + 1):
                label = f"{mode}-run{i}"
                records.append(docker_run(image, output / label, label, flags, args))
        ok = ok and all(r["ok"] for r in records)
    finally:
        if not args.keep_image:
            subprocess.run(["docker", "image", "rm", "-f", image], capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)
        for record in records:
            record.pop("_outcomes", None)
        (output / "container-runs.json").write_text(
            json.dumps(
                {
                    "wheel": wheel.name,
                    "image_removed": not args.keep_image,
                    "ok": ok,
                    "runs": records,
                },
                indent=2,
            )
        )
    print(
        f"[container_run] {'ALL RUNS PASSED' if ok else 'FAILED'}; details in {output / 'container-runs.json'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
