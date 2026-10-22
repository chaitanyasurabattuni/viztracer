#!/usr/bin/env python3
# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/viztracer/blob/master/NOTICE.txt

"""
Release artifact checks used by the GitHub workflows.

  select-wheel DIR          print "wheel=PATH" for the one wheel in DIR that
                            matches the running interpreter
  record-tested WHEEL SUMMARY --out DIR
                            record that WHEEL passed the test matrix whose
                            tools/test_matrix.py summary is SUMMARY
  verify DIST [--tested DIR] [--expect-version V]
                            check the release set: one sdist, wheels, a
                            single version everywhere, every wheel built by
                            the pinned setuptools and (with --tested) every
                            wheel recorded as tested with the same sha256
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent / "requirements"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpreter_tag() -> str:
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        tag += "t"
    return tag


def select_wheel(args: argparse.Namespace) -> int:
    tag = interpreter_tag()
    matches = [w for w in Path(args.dir).glob("*.whl") if w.name.split("-")[3] == tag]
    if len(matches) != 1:
        print(
            f"expected one {tag} wheel in {args.dir}, found {[m.name for m in matches]}",
            file=sys.stderr,
        )
        return 1
    print(f"wheel={matches[0]}")
    return 0


def record_tested(args: argparse.Namespace) -> int:
    wheel = Path(args.wheel)
    summary = json.loads(Path(args.summary).read_text())
    if summary["wheel"] != wheel.name or summary["wheel_sha256"] != sha256(wheel):
        print(f"{args.summary} is not a summary for {wheel}", file=sys.stderr)
        return 1
    if not summary["ok"]:
        print(f"{wheel.name} did not pass the test matrix", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "wheel": wheel.name,
        "sha256": summary["wheel_sha256"],
        "interpreter": summary["interpreter"],
        "runs": [
            {
                k: r[k]
                for k in ("id", "environment", "packages", "passed", "skipped", "ok")
            }
            for r in summary["runs"]
        ],
    }
    (out / f"{wheel.name}.json").write_text(json.dumps(record, indent=2))
    return 0


def pinned_setuptools() -> str:
    match = re.search(
        r"^setuptools==(\S+)", (REQUIREMENTS / "build-backend.txt").read_text(), re.M
    )
    assert match is not None
    return match.group(1)


def verify(args: argparse.Namespace) -> int:
    dist = Path(args.dist)
    errors = []
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    others = [p for p in dist.iterdir() if p not in sdists and p not in wheels]
    if len(sdists) != 1:
        errors.append(f"expected exactly one sdist, found {[s.name for s in sdists]}")
    if not wheels:
        errors.append("no wheels")
    if others:
        errors.append(f"unexpected files: {[p.name for p in others]}")

    versions = {}
    for sdist in sdists:
        with tarfile.open(sdist) as tf:
            member = next(
                m
                for m in tf.getmembers()
                if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
            )
            versions[sdist.name] = email.parser.Parser().parsestr(
                tf.extractfile(member).read().decode()
            )["Version"]

    generator = f"setuptools ({pinned_setuptools()})"
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
        versions[wheel.name] = meta["Version"]
        if wheel.name.split("-")[1] != meta["Version"]:
            errors.append(f"{wheel.name}: file name and metadata versions differ")
        if wheel_meta["Generator"] != generator:
            errors.append(
                f"{wheel.name}: built by {wheel_meta['Generator']}, expected {generator}"
            )

    if len(set(versions.values())) > 1:
        errors.append(f"artifacts carry different versions: {versions}")
    if args.expect_version and set(versions.values()) != {args.expect_version}:
        errors.append(
            f"expected version {args.expect_version}, artifacts carry {sorted(set(versions.values()))}"
        )

    if args.tested:
        tested = {}
        for record in Path(args.tested).glob("*.json"):
            data = json.loads(record.read_text())
            tested[data["wheel"]] = data["sha256"]
        for wheel in wheels:
            if wheel.name not in tested:
                errors.append(
                    f"{wheel.name} was built but never passed the test matrix"
                )
            elif tested[wheel.name] != sha256(wheel):
                errors.append(f"{wheel.name} differs from the wheel that was tested")
        for name in sorted(set(tested) - {w.name for w in wheels}):
            errors.append(f"{name} was tested but is not in the release set")

    for name in sorted(versions):
        print(f"{versions[name]}  {name}")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print(f"verified {len(sdists)} sdist and {len(wheels)} wheels")
    return 1 if errors else 0


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select-wheel")
    p.add_argument("dir")
    p.set_defaults(func=select_wheel)

    p = sub.add_parser("record-tested")
    p.add_argument("wheel")
    p.add_argument("summary")
    p.add_argument("--out", required=True)
    p.set_defaults(func=record_tested)

    p = sub.add_parser("verify")
    p.add_argument("dist")
    p.add_argument("--tested")
    p.add_argument("--expect-version")
    p.set_defaults(func=verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
