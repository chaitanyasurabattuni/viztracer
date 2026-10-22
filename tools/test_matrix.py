#!/usr/bin/env python3
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/viztracer/blob/master/NOTICE.txt

"""
Prepare and run the VizTracer test suite against an installed wheel.

The work is split into two phases so the test phase is hermetic:

  prepare   (network allowed) create one virtual environment per combination
            of the optional packages the suite exercises (orjson, trio),
            install the pinned test requirements and the wheel into each,
            copy the tests out of the source tree and provision Perfetto's
            trace_processor_shell into an isolated HOME.

  run       (no network needed) run the full suite in the "full" environment
            and the package-matrix tests in every other environment, against
            the installed wheel, offline. Writes JUnit XML, a log per run and
            summary.json, and exits non-zero if anything failed.

Only the standard library is used so this works with any interpreter,
inside containers and on CI runners.

    python tools/test_matrix.py prepare --wheel dist/viztracer-...whl --root /tmp/vt
    python tools/test_matrix.py run --root /tmp/vt --results-dir results/
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = Path(__file__).resolve().parent / "requirements"
OPTIONAL_PACKAGES = ("orjson", "trio")
DEFAULT_RUN_TIMEOUT = 30 * 60


def log(msg: str) -> None:
    print(f"[test_matrix] {msg}", flush=True)


def env_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def env_bin(env_dir: Path) -> Path:
    return env_python(env_dir).parent


def run_checked(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def interpreter_info(python: str) -> dict:
    code = (
        "import json, platform, sys, sysconfig;"
        "print(json.dumps({'version': platform.python_version(),"
        "'free_threaded': bool(sysconfig.get_config_var('Py_GIL_DISABLED')),"
        "'executable': sys.executable, 'machine': platform.machine(),"
        "'platform': sys.platform}))"
    )
    out = subprocess.run(
        [python, "-c", code], check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    return json.loads(out)


def environment_plan(free_threaded: bool) -> dict:
    """
    Environments to prepare, as name -> set of optional packages installed.

    "full" gets the full suite, every other environment runs only the tests
    marked package_matrix. Together they cover every present/absent
    combination of orjson and trio. orjson publishes no free-threaded wheels
    and the suite declares no orjson matrix there, so free-threaded builds
    only vary trio.
    """
    if free_threaded:
        return {"full": {"trio"}, "bare": set()}
    return {
        "full": {"orjson", "trio"},
        "orjson-only": {"orjson"},
        "trio-only": {"trio"},
        "bare": set(),
    }


# ---------------------------------------------------------------------------
# trace_processor_shell provisioning
# ---------------------------------------------------------------------------


def load_processor_manifest(wrapper: Path) -> list:
    """Read TRACE_PROCESSOR_SHELL_MANIFEST from viztracer's trace_processor wrapper"""
    tree = ast.parse(wrapper.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TRACE_PROCESSOR_SHELL_MANIFEST"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"no TRACE_PROCESSOR_SHELL_MANIFEST in {wrapper}")


def processor_entry(manifest: list, plat: str, machine: str) -> dict:
    # Same matching rule as the wrapper's get_perfetto_prebuilt()
    for entry in manifest:
        if entry.get("platform") == plat and machine.lower() in entry.get(
            "machine", []
        ):
            return entry
    raise RuntimeError(f"no trace_processor_shell prebuilt for {plat}-{machine}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    if shutil.which("curl"):
        # curl uses the platform trust store, which matters behind TLS proxies
        run_checked(["curl", "-fsSL", "--retry", "3", "-o", tmp, url])
    else:
        import urllib.request

        log(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    os.replace(tmp, dest)


def provision_processor(
    python: Path, home: Path, download_cache: Path | None, work: Path
) -> dict:
    """
    Put the trace_processor_shell that viztracer's wrapper expects into
    HOME/.local/share/perfetto/prebuilts so the wrapper never downloads it.
    """
    wrapper = Path(
        subprocess.run(
            [
                str(python),
                "-c",
                "import os, viztracer; print(os.path.join(os.path.dirname("
                "viztracer.__file__), 'web_dist', 'trace_processor'))",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    )
    entry = processor_entry(
        load_processor_manifest(wrapper), sys.platform, platform.machine()
    )
    cache = download_cache or (work / "downloads")
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{entry['arch']}-{entry['sha256'][:16]}-{entry['file_name']}"
    if not cached.exists() or sha256_file(cached) != entry["sha256"]:
        download(entry["url"], cached)
    actual = sha256_file(cached)
    if actual != entry["sha256"]:
        raise RuntimeError(
            f"checksum mismatch for {entry['url']}: {actual} != {entry['sha256']}"
        )

    prebuilts = home / ".local" / "share" / "perfetto" / "prebuilts"
    prebuilts.mkdir(parents=True, exist_ok=True)
    target = prebuilts / entry["file_name"]
    shutil.copyfile(cached, target)
    os.chmod(target, 0o755)
    (prebuilts / (entry["file_name"] + ".sha256")).write_text(entry["sha256"])

    # The wrapper must now start the binary without touching the network
    result = subprocess.run(
        [str(python), str(wrapper), "--version"],
        env={
            **os.environ,
            "HOME": str(home),
            "USERPROFILE": str(home),
            "PATH": str(env_bin(python.parent.parent)) + os.pathsep + "/nonexistent",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or "Perfetto" not in result.stdout:
        raise RuntimeError(
            f"provisioned trace processor does not run:\n{result.stdout}"
        )
    return {
        "arch": entry["arch"],
        "url": entry["url"],
        "sha256": entry["sha256"],
        "path": str(target),
        "version": result.stdout.strip().splitlines()[0],
    }


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def wheel_version(wheel: Path) -> str:
    return wheel.name.split("-")[1]


def copy_test_sources(source: Path, work: Path) -> None:
    """Copy what the suite needs out of the source tree, never the package itself"""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    shutil.copytree(
        source / "tests",
        work / "tests",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(source / "example" / "json", work / "example" / "json")
    shutil.copyfile(source / ".coveragerc", work / ".coveragerc")


def prepare(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    wheel = Path(args.wheel).resolve()
    source = Path(args.source).resolve()
    requirements = Path(args.requirements).resolve()
    python = args.python or sys.executable
    installer = args.installer or ("uv" if shutil.which("uv") else "pip")
    info = interpreter_info(python)
    plan = environment_plan(info["free_threaded"])
    if args.envs:
        plan = {name: pkgs for name, pkgs in plan.items() if name in args.envs}

    root.mkdir(parents=True, exist_ok=True)
    copy_test_sources(source, root / "work")
    home = root / "home"
    home.mkdir(exist_ok=True)

    manifest = {
        "wheel": wheel.name,
        "wheel_sha256": sha256_file(wheel),
        "version": wheel_version(wheel),
        "interpreter": info,
        "installer": installer,
        "environments": {},
    }

    for name, packages in plan.items():
        env_dir = root / "envs" / name
        if env_dir.exists():
            shutil.rmtree(env_dir)
        log(
            f"preparing environment {name} ({', '.join(sorted(packages)) or 'no optional packages'})"
        )
        py = env_python(env_dir)
        req_args = ["-r", requirements / "test.txt"]
        for pkg in sorted(packages):
            req_args += ["-r", requirements / f"{pkg}.txt"]
        if installer == "uv":
            run_checked(["uv", "venv", "--no-config", "--python", python, env_dir])
            run_checked(
                ["uv", "pip", "install", "--no-config", "--python", py]
                + ["--only-binary", ":all:", *req_args, wheel]
            )
            freeze_cmd = ["uv", "pip", "freeze", "--python", str(py)]
        else:
            run_checked([python, "-m", "venv", env_dir])
            pip = [py, "-m", "pip", "--disable-pip-version-check", "--no-input"]
            run_checked(pip + ["install", "--only-binary=:all:", *req_args, wheel])
            freeze_cmd = [str(py), "-m", "pip", "freeze", "--all"]
        freeze = subprocess.run(
            freeze_cmd, check=True, stdout=subprocess.PIPE, text=True
        ).stdout.split()
        manifest["environments"][name] = {
            "path": str(env_dir),
            "python": str(py),
            "packages": {
                pkg: ("present" if pkg in packages else "absent")
                for pkg in OPTIONAL_PACKAGES
            },
            "suite": "full" if name == "full" else "package_matrix",
            "freeze": freeze,
        }

    first_env = next(iter(manifest["environments"].values()))
    manifest["trace_processor"] = provision_processor(
        Path(first_env["python"]),
        home,
        Path(args.download_cache).resolve() if args.download_cache else None,
        root,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"prepared {len(manifest['environments'])} environments in {root}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def junit_counts(path: Path) -> dict:
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    outcomes = {}
    if not path.exists():
        return {**counts, "outcomes": outcomes}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, 0))
        for case in suite.iter("testcase"):
            name = f"{case.get('classname')}::{case.get('name')}"
            if case.find("failure") is not None:
                outcome = "failed"
            elif case.find("error") is not None:
                outcome = "error"
            elif case.find("skipped") is not None:
                outcome = "skipped"
            else:
                outcome = "passed"
            outcomes[name] = outcome
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    return {**counts, "outcomes": outcomes}


def offline_env(root: Path, env: dict, env_dir: Path) -> dict:
    """Environment for the test phase: isolated HOME, the env's scripts first on PATH, no installers"""
    run_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            ("PYTHON", "PIP_", "UV_", "COVERAGE", "VIRTUAL_ENV", "CONDA")
        )
        and k not in ("VIZTRACER_REPORT_SERVER_ENDPOINT",)
    }
    run_env.update(
        {
            "HOME": str(root / "home"),
            "PATH": str(env_bin(env_dir)) + os.pathsep + os.environ.get("PATH", ""),
            "VIRTUAL_ENV": str(env_dir),
            "VIZTRACER_TEST_PACKAGES": ",".join(
                f"{pkg}={state}" for pkg, state in env["packages"].items()
            ),
            # Anything that tries to install packages fails instead of downloading
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "UV_OFFLINE": "1",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    if sys.platform == "darwin":
        # Keep per-user caches inside the throwaway HOME as well
        run_env["XDG_CACHE_HOME"] = str(root / "home" / ".cache")
    if sys.platform == "win32":
        # os.path.expanduser() reads USERPROFILE on Windows
        run_env["USERPROFILE"] = str(root / "home")
    return run_env


def kill_process_group(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def run_suite(
    root: Path,
    name: str,
    env: dict,
    manifest: dict,
    results: Path,
    label: str,
    coverage: bool,
    timeout: float,
    extra_pytest_args: list,
    keep_runs: bool = False,
) -> dict:
    env_dir = Path(env["path"])
    py = env_python(env_dir)
    run_id = f"{label}-{name}" + ("-coverage" if coverage else "")
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copytree(root / "work" / "tests", run_dir / "tests")
    shutil.copyfile(root / "work" / ".coveragerc", run_dir / ".coveragerc")
    (run_dir / "example").mkdir()
    link_or_copy(root / "work" / "example" / "json", run_dir / "example" / "json")

    run_env = offline_env(root, env, env_dir)
    junit = results / f"{run_id}.xml"
    log_path = results / f"{run_id}.log"
    pytest_args = ["-p", "no:cacheprovider", "-ra", "-v", f"--junit-xml={junit}"]
    pytest_args += ["-o", "junit_family=xunit2", "-o", f"junit_suite_name={run_id}"]
    if env["suite"] == "package_matrix":
        pytest_args += ["-m", "package_matrix"]
    pytest_args += extra_pytest_args
    if coverage:
        run_env["COVERAGE_RUN"] = "True"
        cmd = [
            str(env_bin(env_dir) / "coverage"),
            "run",
            "--source",
            "viztracer",
            "--parallel-mode",
            "-m",
            "pytest",
            *pytest_args,
        ]
    else:
        cmd = [str(py), "-m", "pytest", *pytest_args]

    record = {
        "id": run_id,
        "environment": name,
        "packages": env["packages"],
        "suite": env["suite"],
        "coverage": coverage,
        "command": cmd,
        "log": str(log_path),
        "junit": str(junit),
    }

    with open(log_path, "w") as logf:
        # Confirm the suite imports the installed wheel, not a source tree
        check = subprocess.run(
            [
                str(py),
                "-c",
                "import viztracer, json; print(json.dumps([viztracer.__file__, viztracer.__version__]))",
            ],
            cwd=run_dir,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logf.write(f"$ viztracer import check\n{check.stdout}\n")
        try:
            viz_file, viz_version = json.loads(check.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            viz_file, viz_version = "", ""
        record["viztracer_file"] = viz_file
        record["viztracer_version"] = viz_version
        import_ok = (
            check.returncode == 0
            and viz_version == manifest["version"]
            and Path(viz_file).resolve().is_relative_to(env_dir.resolve())
        )

        logf.write("$ " + " ".join(cmd) + "\n")
        logf.flush()
        start = time.monotonic()
        timed_out = False
        proc = subprocess.Popen(
            cmd,
            cwd=run_dir,
            env=run_env,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=sys.platform != "win32",
        )
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_group(proc)
            returncode = proc.wait()
        # Nothing the suite started may outlive it
        kill_process_group(proc)
        duration = time.monotonic() - start
        if coverage and not timed_out:
            combine = subprocess.run(
                [str(env_bin(env_dir) / "coverage"), "combine"],
                cwd=run_dir,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logf.write(f"$ coverage combine\n{combine.stdout}\n")
            record["coverage_combine_returncode"] = combine.returncode
            if combine.returncode == 0:
                report = subprocess.run(
                    [str(env_bin(env_dir) / "coverage"), "report", "--format=total"],
                    cwd=run_dir,
                    env=run_env,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                record["coverage_total"] = report.stdout.strip()
                xml = results / f"{run_id}.coverage.xml"
                subprocess.run(
                    [str(env_bin(env_dir) / "coverage"), "xml", "-i", "-o", str(xml)],
                    cwd=run_dir,
                    env=run_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if xml.exists():
                    record["coverage_xml"] = str(xml)
        if timed_out:
            logf.write(f"\n[test_matrix] TIMEOUT after {timeout}s, killed\n")

    counts = junit_counts(junit)
    outcomes = counts.pop("outcomes")
    record.update(
        {
            "returncode": returncode,
            "timed_out": timed_out,
            "duration_seconds": round(duration, 1),
            "import_check_ok": import_ok,
            **counts,
        }
    )
    record["ok"] = (
        import_ok
        and not timed_out
        and returncode == 0
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and record.get("coverage_combine_returncode", 0) == 0
    )
    (results / f"{run_id}.outcomes.json").write_text(
        json.dumps(outcomes, indent=1, sort_keys=True)
    )
    record["outcomes"] = str(results / f"{run_id}.outcomes.json")
    if not keep_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
    status = "ok" if record["ok"] else "FAILED"
    log(
        f"{run_id}: {status} rc={returncode} passed={counts['passed']} failed={counts['failures']} "
        f"errors={counts['errors']} skipped={counts['skipped']} in {record['duration_seconds']}s"
    )
    return record


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    results = Path(args.results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    label = args.label or re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        manifest["wheel"].split("-")[2] + "-" + manifest["wheel"].split("-")[3],
    )
    envs = manifest["environments"]
    if args.envs:
        envs = {name: env for name, env in envs.items() if name in args.envs}

    runs = []
    for iteration in range(1, args.repeat + 1):
        for name, env in envs.items():
            this_label = label if args.repeat == 1 else f"{label}-run{iteration}"
            runs.append(
                run_suite(
                    root,
                    name,
                    env,
                    manifest,
                    results,
                    this_label,
                    args.coverage,
                    args.timeout,
                    args.pytest_args,
                    args.keep_runs,
                )
            )
            if args.fail_fast and not runs[-1]["ok"]:
                break
        if args.fail_fast and runs and not runs[-1]["ok"]:
            break

    summary = {
        "wheel": manifest["wheel"],
        "wheel_sha256": manifest["wheel_sha256"],
        "version": manifest["version"],
        "interpreter": manifest["interpreter"],
        "trace_processor": manifest["trace_processor"],
        "environments": {
            name: {k: v for k, v in env.items() if k != "path"}
            for name, env in manifest["environments"].items()
        },
        "runs": runs,
        "ok": bool(runs) and all(r["ok"] for r in runs),
    }
    summary_path = results / f"summary-{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.chown:
        uid, gid = (int(x) for x in args.chown.split(":"))
        for path in [results, *results.rglob("*")]:
            try:
                os.chown(path, uid, gid)
            except OSError:
                pass
    log(f"summary written to {summary_path}: {'ok' if summary['ok'] else 'FAILED'}")
    return 0 if summary["ok"] else 1


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="create the test environments (network allowed)")
    p.add_argument("--wheel", required=True, help="wheel to install")
    p.add_argument(
        "--root", required=True, help="directory to create the environments in"
    )
    p.add_argument(
        "--python", help="interpreter matching the wheel (default: this one)"
    )
    p.add_argument(
        "--source",
        default=str(SOURCE_ROOT),
        help="source checkout to copy the tests from",
    )
    p.add_argument(
        "--requirements",
        default=str(REQUIREMENTS),
        help="directory with the pinned requirement files",
    )
    p.add_argument(
        "--download-cache", help="directory to cache the trace processor download in"
    )
    p.add_argument("--envs", nargs="*", help="only prepare these environments")
    p.add_argument(
        "--installer",
        choices=["pip", "uv"],
        help="how to create environments (default: uv if on PATH, else pip)",
    )
    p.set_defaults(func=prepare)

    r = sub.add_parser("run", help="run the suite in prepared environments (offline)")
    r.add_argument("--root", required=True, help="directory given to prepare")
    r.add_argument(
        "--results-dir",
        required=True,
        help="where to write logs, JUnit XML and summary",
    )
    r.add_argument(
        "--label", help="prefix for result files (default: from the wheel tags)"
    )
    r.add_argument(
        "--coverage", action="store_true", help="run the way CI measures coverage"
    )
    r.add_argument(
        "--repeat", type=int, default=1, help="run everything this many times"
    )
    r.add_argument("--envs", nargs="*", help="only run in these environments")
    r.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_RUN_TIMEOUT,
        help="seconds before a run is killed",
    )
    r.add_argument(
        "--fail-fast", action="store_true", help="stop after the first failed run"
    )
    r.add_argument(
        "--keep-runs", action="store_true", help="keep the per-run working directories"
    )
    r.add_argument(
        "--chown", help="UID:GID to give the results to (for container runs)"
    )
    r.add_argument("pytest_args", nargs="*", help="extra pytest arguments (after --)")
    r.set_defaults(func=run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
