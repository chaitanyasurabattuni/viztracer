# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/gaogaotiantian/viztracer/blob/master/NOTICE.txt


import sys
import unittest

from .cmdline_tmpl import CmdlineTmpl
from .package_env import package_available, package_matrix

TORCH_MISSING_REASON = "torch is not installed in this environment"


def check_torch_events(test, data):
    events = data["traceEvents"]
    py_events = [e for e in events if e["name"] == "torch.empty"]
    aten_events = [e for e in events if e["name"] == "aten::empty"]
    test.assertEqual(len(py_events), 100)
    test.assertEqual(len(aten_events), 100)
    for py, aten in zip(py_events, aten_events):
        if "linux" in sys.platform:
            # We care about Linux
            test.assertLess(py["ts"], aten["ts"])
            test.assertGreater(py["ts"] + py["dur"], aten["ts"] + aten["dur"])
        elif sys.platform == "win32":
            # Windows is at least sane, give it 100us diff
            acceptable_margin = 100
            test.assertLess(py["ts"], aten["ts"] + acceptable_margin)
            test.assertGreater(
                py["ts"] + py["dur"],
                aten["ts"] + aten["dur"] - acceptable_margin,
            )
        else:
            # Mac is pure crazy and we don't care about it
            pass


@package_matrix(["~torch", "torch"])
class TestTorch(CmdlineTmpl):
    @unittest.skipUnless(package_available("torch"), TORCH_MISSING_REASON)
    def test_basic(self):
        script = """
            import torch
            from viztracer import VizTracer
            with VizTracer(log_torch=True, verbose=0):
                for i in range(100):
                    torch.empty(i)
        """

        self.template(
            [sys.executable, "cmdline_test.py"],
            script=script,
            check_func=lambda data: check_torch_events(self, data),
        )

    @unittest.skipUnless(package_available("torch"), TORCH_MISSING_REASON)
    def test_cmdline(self):
        script = """
            import torch
            for i in range(100):
                torch.empty(i)
        """

        self.template(
            ["viztracer", "--log_torch", "cmdline_test.py"],
            script=script,
            check_func=lambda data: check_torch_events(self, data),
        )

    @unittest.skipIf(package_available("torch"), "torch is installed")
    def test_log_torch_without_torch(self):
        script = """
            from viztracer import VizTracer
            _ = VizTracer(log_torch=True, verbose=0)
        """
        self.template(
            [sys.executable, "cmdline_test.py"],
            script=script,
            expected_output_file=None,
            success=False,
            expected_stderr=".*ModuleNotFoundError.*",
        )
        self.template(
            ["viztracer", "--log_torch", "cmdline_test.py"],
            script="pass",
            success=False,
        )
