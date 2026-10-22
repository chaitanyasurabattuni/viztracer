# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/viztracer/blob/master/NOTICE.txt

"""
Package matrix support for tests whose behavior depends on optional packages.

Tests never install or uninstall packages. A test decorated with
``package_matrix`` runs once against whatever the current environment provides,
and is marked with the ``package_matrix`` pytest marker. The release gate
(tools/test_matrix.py) prepares one environment per combination of the optional
packages before the test phase and runs the marked tests in each of them.
"""

import importlib.util
import inspect

import pytest

# Optional packages that change viztracer's behavior
OPTIONAL_PACKAGES = ("orjson", "trio", "torch")


def package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


class PackageConfig:
    def __init__(self, packages):
        self.packages = frozenset(packages)

    def has(self, package):
        return package in self.packages

    def __repr__(self):
        return f"PackageConfig({sorted(self.packages)})"


def current_package_config(packages=OPTIONAL_PACKAGES) -> PackageConfig:
    return PackageConfig(pkg for pkg in packages if package_available(pkg))


def _packages_in_matrix(pkg_matrix):
    names = set()
    for entry in pkg_matrix:
        for pkg in entry if isinstance(entry, list) else [entry]:
            names.add(pkg.lstrip("~"))
    return sorted(names)


def package_matrix(pkg_matrix):
    """
    pkg_matrix lists the variants a test supports, e.g. ["~orjson", "orjson"]
    for "without orjson" and "with orjson". None or [] means the test does not
    vary with optional packages here.
    """

    def inner(func_or_cls):
        if not pkg_matrix:
            return func_or_cls
        packages = _packages_in_matrix(pkg_matrix)
        marked = pytest.mark.package_matrix(packages=packages)(func_or_cls)
        if inspect.isclass(marked):
            marked.pkg_config = current_package_config(packages)
        return marked

    return inner
