#!/usr/bin/env python3
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/viztracer/blob/master/NOTICE.txt

"""
Release gate: build every release artifact from this checkout, install each
wheel into a fresh environment outside the source tree, run the full test
suite against it offline, and fail if anything fails.

    python tools/release_gate.py --output-dir PATH

Builds, on a macOS arm64 host with uv and Docker:
  * one sdist
  * cp312, cp314 and cp314t macOS arm64 wheels, from the sdist, with
    uv-managed interpreters
  * cp312, cp314 and cp314t manylinux aarch64 wheels, from the sdist, with
    cibuildwheel

Then, for every wheel, prepares the test environments with
tools/test_matrix.py (one per orjson/trio combination, with Perfetto's
trace_processor_shell provisioned) and runs the suite: in a venv on macOS,
and in a python:*-slim container with --network none on Linux.

PATH receives results.json, the artifacts (dist/), a log per stage (logs/)
and per-run logs and JUnit XML (tests/). Temporary environments and images
are removed whether the gate passes or fails. See RELEASING.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import email.parser
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = SOURCE_ROOT / "tools" / "requirements"

# Pinned tool and environment versions. Bump deliberately.
UV_VERSION = "0.11.2"
TOOLS_PYTHON = "3.12.13"
# wheel tag -> uv-managed interpreter used for macOS builds and tests
MACOS_PYTHONS = {"cp312": "3.12.13", "cp314": "3.14.3", "cp314t": "3.14.3t"}
# wheel tag -> (base image, uv-managed interpreter or None for the image's own)
LINUX_TEST_IMAGES = {
    "cp312": (
        "python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2",
        None,
    ),
    "cp314": (
        "python:3.14-slim@sha256:af30dd6e5f99a127a05c5c1189f0f454b389c87d6ad9ffbab7a8cef12d3852e5",
        None,
    ),
    "cp314t": (
        "python:3.14-slim@sha256:af30dd6e5f99a127a05c5c1189f0f454b389c87d6ad9ffbab7a8cef12d3852e5",
        "3.14.3t",
    ),
}
WHEEL_TAGS = ("cp312", "cp314", "cp314t")
CIBW_BUILD = " ".join(f"{tag}-manylinux_aarch64" for tag in WHEEL_TAGS)
RUN_TIMEOUT = 30 * 60


class GateError(Exception):
    pass


class Gate:
    def __init__(self, output_dir: Path, jobs: int) -> None:
        self.output_dir = output_dir
        self.jobs = jobs
        self.dist = output_dir / "dist"
        self.logs = output_dir / "logs"
        self.tests = output_dir / "tests"
        self.run_id = uuid.uuid4().hex[:8]
        self.tmp = Path(tempfile.mkdtemp(prefix=f"viztracer-gate-{self.run_id}-"))
        self.images: list[str] = []
        self.containers: list[str] = []
        self.stage_counter = 0
        self.lock = threading.Lock()
        self.results: dict = {
            "schema": 1,
            "ok": False,
            "started": now(),
            "finished": None,
            "output_dir": str(output_dir),
            "source": {},
            "tools": {},
            "version": None,
            "artifacts": [],
            "environments": [],
            "test_runs": [],
            "stages": [],
            "failures": [],
        }
        self.env = dict(os.environ)
        # uv must trust the platform certificate store (TLS-inspecting proxies)
        self.env.setdefault("UV_SYSTEM_CERTS", "1")
        self.env["UV_PYTHON_PREFERENCE"] = "only-managed"
        for var in ("PYTHONPATH", "VIRTUAL_ENV", "PIP_REQUIRE_VIRTUALENV"):
            self.env.pop(var, None)

    # -- helpers ------------------------------------------------------------

    def say(self, msg: str) -> None:
        with self.lock:
            print(f"[gate {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def new_log(self, name: str) -> Path:
        with self.lock:
            self.stage_counter += 1
            return self.logs / f"{self.stage_counter:02d}-{name}.log"

    def run(
        self,
        cmd: list,
        log: Path,
        cwd: Path | None = None,
        env: dict | None = None,
        timeout: float | None = None,
    ) -> str:
        """Run a command, append its output to log, raise GateError on failure"""
        cmd = [str(c) for c in cmd]
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env or self.env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as e:
                f.write(e.output or "")
                raise GateError(
                    f"timed out after {timeout}s: {' '.join(cmd)}"
                ) from None
            except FileNotFoundError as e:
                raise GateError(f"command not found: {e.filename}") from None
            f.write(proc.stdout)
        if proc.returncode != 0:
            raise GateError(f"exit code {proc.returncode}: {' '.join(cmd)} (see {log})")
        return proc.stdout

    def stage(self, name: str, func, *args):
        log = self.new_log(name)
        record = {"name": name, "status": "running", "log": rel(log, self.output_dir)}
        with self.lock:
            self.results["stages"].append(record)
        self.say(f"{name} ...")
        start = time.monotonic()
        try:
            value = func(log, *args)
        except Exception as e:
            record["status"] = "failed"
            record["error"] = str(e)
            with open(log, "a") as f:
                f.write("\n" + traceback.format_exc())
            self.say(f"{name} FAILED: {e}")
            raise
        finally:
            record["duration_seconds"] = round(time.monotonic() - start, 1)
        record["status"] = "passed"
        self.say(f"{name} ok ({record['duration_seconds']}s)")
        return value

    # -- stages -------------------------------------------------------------

    def preflight(self, log: Path) -> None:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise GateError(
                "the release gate builds macOS arm64 wheels and must run on a macOS arm64 host"
            )
        uv = self.run(["uv", "--version"], log).split()[1]
        if uv != UV_VERSION:
            raise GateError(f"uv {UV_VERSION} is required, found {uv}")
        docker = self.run(
            ["docker", "version", "--format", "{{.Server.Version}} {{.Server.Arch}}"],
            log,
        ).strip()
        if not docker.endswith(("arm64", "aarch64")):
            raise GateError(
                f"Docker must run linux/arm64 containers natively, found {docker}"
            )
        self.results["tools"].update({"uv": uv, "docker_server": docker})

        source = {"path": str(SOURCE_ROOT)}
        if (SOURCE_ROOT / ".git").exists():
            source["git_commit"] = self.run(
                ["git", "-C", SOURCE_ROOT, "rev-parse", "HEAD"], log
            ).strip()
            status = self.run(["git", "-C", SOURCE_ROOT, "status", "--porcelain"], log)
            source["git_dirty"] = bool(status.strip())
        self.results["source"] = source

    def export_source(self, log: Path) -> Path:
        """Copy the checkout so builds never write into the source tree"""
        dest = self.tmp / "source"
        if (SOURCE_ROOT / ".git").exists():
            files = self.run(
                [
                    "git",
                    "-C",
                    SOURCE_ROOT,
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                log,
            ).split("\0")
            for name in filter(None, files):
                src = SOURCE_ROOT / name
                if src.is_file():
                    (dest / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest / name)
        else:
            shutil.copytree(
                SOURCE_ROOT,
                dest,
                ignore=shutil.ignore_patterns(
                    ".git", "build", "dist", "*.egg-info", "__pycache__"
                ),
            )
        return dest

    def tools_env(self, log: Path) -> Path:
        self.run(
            ["uv", "python", "install", TOOLS_PYTHON, *MACOS_PYTHONS.values()], log
        )
        venv = self.tmp / "tools-venv"
        self.run(["uv", "venv", "--python", TOOLS_PYTHON, venv], log)
        self.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                venv / "bin" / "python",
                "-r",
                REQUIREMENTS / "tools.txt",
            ],
            log,
        )
        versions = self.run(
            [
                venv / "bin" / "python",
                "-c",
                "import importlib.metadata as m; print(m.version('cibuildwheel'), m.version('twine'))",
            ],
            log,
        ).split()
        self.results["tools"]["cibuildwheel"], self.results["tools"]["twine"] = versions
        self.results["tools"]["setuptools"] = pinned(
            "setuptools", REQUIREMENTS / "build-backend.txt"
        )
        return venv

    def build_sdist(self, log: Path, source: Path) -> Path:
        self.run(
            [
                "uv",
                "build",
                "--sdist",
                "--python",
                TOOLS_PYTHON,
                "--no-config",
                "--build-constraint",
                REQUIREMENTS / "build-backend.txt",
                "--out-dir",
                self.dist,
                source,
            ],
            log,
        )
        sdists = list(self.dist.glob("*.tar.gz"))
        if len(sdists) != 1:
            raise GateError(f"expected one sdist, found {sdists}")
        return sdists[0]

    def build_macos_wheel(self, log: Path, sdist: Path, tag: str) -> Path:
        out = self.tmp / f"wheel-macos-{tag}"
        self.run(
            [
                "uv",
                "build",
                "--wheel",
                "--python",
                MACOS_PYTHONS[tag],
                "--no-config",
                "--build-constraint",
                REQUIREMENTS / "build-backend.txt",
                "--out-dir",
                out,
                sdist,
            ],
            log,
        )
        wheels = list(out.glob(f"*-{tag.rstrip('t')}-{tag}-macosx_*_arm64.whl"))
        if len(wheels) != 1:
            raise GateError(
                f"expected one {tag} macOS wheel, found {list(out.iterdir())}"
            )
        shutil.copy2(wheels[0], self.dist / wheels[0].name)
        return self.dist / wheels[0].name

    def build_linux_wheels(self, log: Path, sdist: Path, tools: Path) -> list[Path]:
        out = self.tmp / "wheel-linux"
        backend_pins = (REQUIREMENTS / "build-backend.txt").read_text()
        pins = " ".join(
            line.split()[0]
            for line in backend_pins.splitlines()
            if line and not line.startswith("#")
        )
        env = {
            **self.env,
            "CIBW_BUILD": CIBW_BUILD,
            "CIBW_ARCHS_LINUX": "aarch64",
            "CIBW_TEST_SKIP": "*",
            "CIBW_CONTAINER_ENGINE": "docker; disable_host_mount: true",
            # Pin the build backend inside the build containers
            "CIBW_BEFORE_ALL_LINUX": f"printf '%s\\n' {pins} > /tmp/viztracer-build-constraints.txt",
            "CIBW_ENVIRONMENT_LINUX": "PIP_CONSTRAINT=/tmp/viztracer-build-constraints.txt PIP_BUILD_CONSTRAINT=/tmp/viztracer-build-constraints.txt",
        }
        self.run(
            [
                tools / "bin" / "cibuildwheel",
                "--platform",
                "linux",
                "--output-dir",
                out,
                sdist,
            ],
            log,
            env=env,
        )
        wheels = sorted(out.glob("*.whl"))
        if len(wheels) != len(WHEEL_TAGS):
            raise GateError(
                f"expected {len(WHEEL_TAGS)} Linux wheels, found {[w.name for w in wheels]}"
            )
        result = []
        for wheel in wheels:
            shutil.copy2(wheel, self.dist / wheel.name)
            result.append(self.dist / wheel.name)
        return result

    def check_artifacts(
        self, log: Path, tools: Path, sdist: Path, wheels: list[Path]
    ) -> None:
        self.run([tools / "bin" / "twine", "check", "--strict", sdist, *wheels], log)

        expected_generator = (
            f"setuptools ({pinned('setuptools', REQUIREMENTS / 'build-backend.txt')})"
        )
        versions = set()
        artifacts = []
        with tarfile.open(sdist) as tf:
            pkg_info = next(
                m
                for m in tf.getmembers()
                if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
            )
            meta = email.parser.Parser().parsestr(
                tf.extractfile(pkg_info).read().decode()
            )
        versions.add(meta["Version"])
        artifacts.append(artifact_record(sdist, "sdist", meta["Version"]))

        seen = set()
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as zf:
                dist_info = next(
                    n.split("/")[0]
                    for n in zf.namelist()
                    if n.endswith(".dist-info/METADATA")
                )
                meta = email.parser.Parser().parsestr(
                    zf.read(f"{dist_info}/METADATA").decode()
                )
                wheel_meta = email.parser.Parser().parsestr(
                    zf.read(f"{dist_info}/WHEEL").decode()
                )
            versions.add(meta["Version"])
            if wheel.name.split("-")[1] != meta["Version"]:
                raise GateError(f"{wheel.name}: file name and metadata versions differ")
            generator = wheel_meta["Generator"]
            if generator != expected_generator:
                raise GateError(
                    f"{wheel.name} was generated by {generator}, expected {expected_generator}"
                )
            tag = wheel.name.split("-")[3]
            plat = "macos-arm64" if "macosx" in wheel.name else "linux-aarch64"
            seen.add((plat, tag))
            record = artifact_record(wheel, "wheel", meta["Version"])
            record.update({"platform": plat, "python_tag": tag, "generator": generator})
            artifacts.append(record)
        if sdist.name.split("-")[-1].removesuffix(".tar.gz") not in versions:
            raise GateError(f"{sdist.name}: file name and metadata versions differ")

        missing = {
            (p, t) for p in ("macos-arm64", "linux-aarch64") for t in WHEEL_TAGS
        } - seen
        if missing or len(wheels) != 2 * len(WHEEL_TAGS):
            raise GateError(f"unexpected wheel set, missing {sorted(missing)}")
        if len(versions) != 1:
            raise GateError(f"artifacts carry different versions: {sorted(versions)}")
        self.results["version"] = versions.pop()
        self.results["artifacts"] = artifacts
        with open(log, "a") as f:
            json.dump(artifacts, f, indent=2)

    def download_processors(self, log: Path) -> Path:
        """Fetch trace_processor_shell for both test platforms once, verified against viztracer's manifest"""
        cache = self.tmp / "downloads"
        cache.mkdir()
        sys.path.insert(0, str(SOURCE_ROOT / "tools"))
        import test_matrix

        manifest = test_matrix.load_processor_manifest(
            SOURCE_ROOT / "src" / "viztracer" / "web_dist" / "trace_processor"
        )
        for plat, machine in (("darwin", "arm64"), ("linux", "aarch64")):
            entry = test_matrix.processor_entry(manifest, plat, machine)
            dest = (
                cache / f"{entry['arch']}-{entry['sha256'][:16]}-{entry['file_name']}"
            )
            self.run(["curl", "-fsSL", "--retry", "3", "-o", dest, entry["url"]], log)
            if test_matrix.sha256_file(dest) != entry["sha256"]:
                raise GateError(f"checksum mismatch for {entry['url']}")
        return cache

    def prepare_macos(
        self, log: Path, wheel: Path, tag: str, source: Path, downloads: Path
    ) -> Path:
        root = self.tmp / f"env-macos-{tag}"
        python = self.run(["uv", "python", "find", MACOS_PYTHONS[tag]], log).strip()
        self.run(
            [
                python,
                source / "tools" / "test_matrix.py",
                "prepare",
                "--wheel",
                wheel,
                "--python",
                python,
                "--root",
                root,
                "--source",
                source,
                "--requirements",
                source / "tools" / "requirements",
                "--download-cache",
                downloads,
                "--installer",
                "uv",
            ],
            log,
        )
        self.add_environment(
            f"macos-arm64-{tag}",
            "macos-arm64",
            wheel,
            root / "manifest.json",
            {"interpreter_request": MACOS_PYTHONS[tag]},
        )
        return root

    def prepare_linux(
        self, log: Path, wheel: Path, tag: str, source: Path, downloads: Path
    ) -> str:
        base, uv_python = LINUX_TEST_IMAGES[tag]
        image = f"viztracer-gate-{self.run_id}-{tag}"
        context = build_image_context(
            self.tmp / f"image-{tag}", source, wheel, downloads, base, uv_python
        )
        self.images.append(image)
        self.run(
            [
                "docker",
                "build",
                "--progress",
                "plain",
                "--network",
                "default",
                "-t",
                image,
                context,
            ],
            log,
        )
        manifest = self.tmp / f"image-{tag}-manifest.json"
        manifest.write_text(
            self.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    image,
                    "cat",
                    "/opt/vt/root/manifest.json",
                ],
                log,
            )
        )
        self.add_environment(
            f"linux-aarch64-{tag}",
            "linux-aarch64",
            wheel,
            manifest,
            {"base_image": base, "uv_python": uv_python, "image": image},
        )
        return image

    def add_environment(
        self, env_id: str, plat: str, wheel: Path, manifest_path: Path, extra: dict
    ) -> None:
        manifest = json.loads(manifest_path.read_text())
        record = {
            "id": env_id,
            "platform": plat,
            "wheel": wheel.name,
            "interpreter": manifest["interpreter"],
            "trace_processor": manifest["trace_processor"],
            "environments": {
                name: {
                    "packages": env["packages"],
                    "suite": env["suite"],
                    "freeze": env["freeze"],
                }
                for name, env in manifest["environments"].items()
            },
            **extra,
        }
        with self.lock:
            self.results["environments"].append(record)

    def test_macos(self, log: Path, root: Path, tag: str, source: Path) -> None:
        label = f"macos-arm64-{tag}"
        results = self.tests / label
        python = self.run(["uv", "python", "find", MACOS_PYTHONS[tag]], log).strip()
        try:
            self.run(
                [
                    python,
                    source / "tools" / "test_matrix.py",
                    "run",
                    "--root",
                    root,
                    "--results-dir",
                    results,
                    "--label",
                    label,
                    "--timeout",
                    RUN_TIMEOUT,
                ],
                log,
                timeout=RUN_TIMEOUT * 5,
            )
        finally:
            self.collect_runs(results, label)

    def test_linux(self, log: Path, image: str, tag: str) -> None:
        label = f"linux-aarch64-{tag}"
        results = self.tests / label
        results.mkdir(parents=True, exist_ok=True)
        container = f"{image}-test"
        self.containers.append(container)
        try:
            self.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    container,
                    "--network",
                    "none",
                    "-v",
                    f"{results}:/results",
                    image,
                    "python",
                    "/opt/vt/src/tools/test_matrix.py",
                    "run",
                    "--root",
                    "/opt/vt/root",
                    "--results-dir",
                    "/results",
                    "--label",
                    label,
                    "--timeout",
                    RUN_TIMEOUT,
                    "--chown",
                    f"{os.getuid()}:{os.getgid()}",
                ],
                log,
                timeout=RUN_TIMEOUT * 5,
            )
        finally:
            self.collect_runs(results, label)

    def collect_runs(self, results: Path, label: str) -> None:
        summary = results / f"summary-{label}.json"
        if not summary.exists():
            with self.lock:
                self.results["test_runs"].append(
                    {"id": label, "ok": False, "error": "no summary written"}
                )
            return
        data = json.loads(summary.read_text())
        with self.lock:
            for record in data["runs"]:
                for key in ("log", "junit", "outcomes"):
                    record[key] = rel(results / Path(record[key]).name, self.output_dir)
                record["environment_id"] = label
                record["wheel"] = data["wheel"]
                self.results["test_runs"].append(record)

    # -- orchestration ------------------------------------------------------

    def execute(self) -> bool:
        for d in (self.dist, self.logs, self.tests):
            d.mkdir(parents=True)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs)
        try:
            self.stage("preflight", self.preflight)
            source = self.stage("export-source", self.export_source)
            tools = self.stage("tools", self.tools_env)
            sdist = self.stage("build-sdist", self.build_sdist, source)
            macos_wheels = {
                tag: self.stage(
                    f"build-macos-{tag}", self.build_macos_wheel, sdist, tag
                )
                for tag in WHEEL_TAGS
            }
            linux_list = self.stage(
                "build-linux-cibuildwheel", self.build_linux_wheels, sdist, tools
            )
            linux_wheels = {w.name.split("-")[3]: w for w in linux_list}
            self.stage(
                "check-artifacts",
                self.check_artifacts,
                tools,
                sdist,
                [*macos_wheels.values(), *linux_list],
            )
            downloads = self.stage("download-trace-processor", self.download_processors)

            # Prepare (network allowed): Linux images build in the background
            linux_prep = {
                tag: pool.submit(
                    self.stage,
                    f"prepare-linux-{tag}",
                    self.prepare_linux,
                    linux_wheels[tag],
                    tag,
                    source,
                    downloads,
                )
                for tag in WHEEL_TAGS
            }
            macos_roots = {
                tag: self.stage(
                    f"prepare-macos-{tag}",
                    self.prepare_macos,
                    macos_wheels[tag],
                    tag,
                    source,
                    downloads,
                )
                for tag in WHEEL_TAGS
            }
            images = {tag: future.result() for tag, future in linux_prep.items()}

            # Test (offline): every wheel, failures collected rather than stopping early
            failures = []
            linux_tests = {
                tag: pool.submit(
                    self.stage, f"test-linux-{tag}", self.test_linux, images[tag], tag
                )
                for tag in WHEEL_TAGS
            }
            for tag in WHEEL_TAGS:
                try:
                    self.stage(
                        f"test-macos-{tag}",
                        self.test_macos,
                        macos_roots[tag],
                        tag,
                        source,
                    )
                except Exception as e:
                    failures.append(f"macos-arm64-{tag}: {e}")
            for tag, future in linux_tests.items():
                try:
                    future.result()
                except Exception as e:
                    failures.append(f"linux-aarch64-{tag}: {e}")

            expected_runs = {
                (env["id"], name)
                for env in self.results["environments"]
                for name in env["environments"]
            }
            ran = {
                (r.get("environment_id"), r.get("environment"))
                for r in self.results["test_runs"]
                if r.get("ok")
            }
            if expected_runs - ran:
                failures.append(
                    f"environments without a passing run: {sorted(expected_runs - ran)}"
                )
            if failures:
                raise GateError("; ".join(failures))
            self.results["ok"] = True
        except BaseException as e:
            self.results["failures"].append(str(e) or type(e).__name__)
        finally:
            # Background builds must finish before their images can be removed
            pool.shutdown(wait=True, cancel_futures=True)
            self.cleanup()
            self.results["finished"] = now()
            (self.output_dir / "results.json").write_text(
                json.dumps(self.results, indent=2)
            )
        return self.results["ok"]

    def cleanup(self) -> None:
        log = self.logs / "cleanup.log"
        for container in self.containers:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        for image in self.images:
            try:
                self.run(["docker", "image", "rm", "-f", image], log)
            except GateError:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.results["cleanup"] = {
            "temporary_directory": str(self.tmp),
            "temporary_directory_removed": not self.tmp.exists(),
            "images_removed": self.images,
        }


def build_image_context(
    context: Path,
    source: Path,
    wheel: Path,
    downloads: Path,
    base: str,
    uv_python: str | None,
) -> Path:
    """Docker build context for a Linux test image: tests, test tooling, the wheel and the processor download"""
    if context.exists():
        shutil.rmtree(context)
    (context / "src" / "tools").mkdir(parents=True)
    shutil.copytree(
        source / "tests",
        context / "src" / "tests",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(source / "example" / "json", context / "src" / "example" / "json")
    shutil.copy2(source / ".coveragerc", context / "src" / ".coveragerc")
    shutil.copy2(
        source / "tools" / "test_matrix.py",
        context / "src" / "tools" / "test_matrix.py",
    )
    shutil.copytree(
        source / "tools" / "requirements", context / "src" / "tools" / "requirements"
    )
    (context / "wheels").mkdir()
    shutil.copy2(wheel, context / "wheels" / wheel.name)
    shutil.copytree(downloads, context / "downloads")

    lines = [
        f"FROM {base}",
        "ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore",
        "COPY src/ /opt/vt/src/",
        "COPY wheels/ /opt/vt/wheels/",
        "COPY downloads/ /opt/vt/downloads/",
    ]
    if uv_python:
        lines += [
            f"RUN python -m pip install uv=={UV_VERSION} \\",
            f" && UV_PYTHON_INSTALL_DIR=/opt/vt/python uv python install {uv_python} \\",
            f' && ln -s "$(UV_PYTHON_INSTALL_DIR=/opt/vt/python uv python find --managed-python {uv_python})" /opt/vt/target-python',
        ]
    else:
        lines.append("RUN ln -s /usr/local/bin/python /opt/vt/target-python")
    lines += [
        "RUN python /opt/vt/src/tools/test_matrix.py prepare \\",
        f"    --wheel /opt/vt/wheels/{wheel.name} --python /opt/vt/target-python \\",
        "    --root /opt/vt/root --source /opt/vt/src --requirements /opt/vt/src/tools/requirements \\",
        "    --download-cache /opt/vt/downloads",
        "WORKDIR /opt/vt",
    ]
    (context / "Dockerfile").write_text("\n".join(lines) + "\n")
    return context


def artifact_record(path: Path, kind: str, version: str) -> dict:
    return {
        "file": path.name,
        "path": f"dist/{path.name}",
        "kind": kind,
        "version": version,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def pinned(package: str, requirements: Path) -> str:
    match = re.search(
        rf"^{re.escape(package)}==(\S+)", requirements.read_text(), re.MULTILINE
    )
    if match is None:
        raise GateError(f"{package} is not pinned in {requirements}")
    return match.group(1)


def rel(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="where results.json, artifacts and logs go (must be empty or absent)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Linux images prepared/tested in parallel (default 3)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"error: {output_dir} is not empty", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    gate = Gate(output_dir, max(1, args.jobs))
    ok = gate.execute()
    runs = gate.results["test_runs"]
    print()
    print(f"version: {gate.results['version']}")
    for artifact in gate.results["artifacts"]:
        print(f"artifact: {artifact['file']}")
    for record in runs:
        status = "ok" if record.get("ok") else "FAILED"
        print(
            f"run {status:6} {record.get('id')}: passed={record.get('passed')} failed={record.get('failures')} "
            f"errors={record.get('errors')} skipped={record.get('skipped')} ({record.get('duration_seconds')}s)"
        )
    for failure in gate.results["failures"]:
        print(f"FAILURE: {failure}")
    print(f"results: {output_dir / 'results.json'}")
    print("RELEASE GATE PASSED" if ok else "RELEASE GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
