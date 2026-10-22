# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/viztracer/blob/master/NOTICE.txt

import os
import signal

import pytest

from .package_env import OPTIONAL_PACKAGES, package_available


def pytest_configure(config):
    # Several tests stop their subprocesses with SIGINT. A shell starts
    # background jobs with SIGINT ignored, children inherit SIG_IGN across
    # exec, and Python then never installs its KeyboardInterrupt handler, so
    # those subprocesses would never stop. Give children the default again.
    if hasattr(signal, "SIGINT") and signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)

    config.addinivalue_line(
        "markers",
        "package_matrix(packages): behavior depends on the optional packages "
        "listed; run in every prepared package combination",
    )

    # tools/test_matrix.py declares which optional packages each prepared
    # environment provides, e.g. "orjson=present,trio=absent". Refuse to run
    # in an environment that does not match, so a combination is never
    # silently skipped.
    expected = os.getenv("VIZTRACER_TEST_PACKAGES")
    if expected:
        mismatches = []
        for item in expected.split(","):
            name, _, state = item.strip().partition("=")
            if state not in ("present", "absent"):
                raise pytest.UsageError(f"bad VIZTRACER_TEST_PACKAGES entry {item!r}")
            if package_available(name) != (state == "present"):
                mismatches.append(f"{name} should be {state}")
        if mismatches:
            raise pytest.UsageError(
                "test environment does not match VIZTRACER_TEST_PACKAGES: "
                + ", ".join(mismatches)
            )


def pytest_report_header(config):
    import viztracer

    packages = ", ".join(
        f"{pkg}={'present' if package_available(pkg) else 'absent'}"
        for pkg in OPTIONAL_PACKAGES
    )
    return [
        f"viztracer {viztracer.__version__} from {os.path.dirname(viztracer.__file__)}",
        f"optional packages: {packages}",
    ]
