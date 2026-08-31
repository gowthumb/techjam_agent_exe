import unittest
from unittest.mock import MagicMock, patch

from agent.dependencies import install, missing_module


class MissingModuleTest(unittest.TestCase):
    def test_extracts_top_level_module_from_a_real_traceback(self):
        trace = (
            "Traceback (most recent call last):\n"
            "  File \"candidate.py\", line 3, in <module>\n"
            "    from catboost import CatBoostRanker\n"
            "ModuleNotFoundError: No module named 'catboost'\n"
        )
        self.assertEqual(missing_module(trace), "catboost")

    def test_extracts_top_level_name_from_a_dotted_submodule_import(self):
        trace = "ModuleNotFoundError: No module named 'sklearn.ensemble'"
        self.assertEqual(missing_module(trace), "sklearn")

    def test_returns_none_for_an_unrelated_error(self):
        self.assertIsNone(missing_module("ValueError: bad shape"))
        self.assertIsNone(missing_module(None))
        self.assertIsNone(missing_module(""))


class InstallTest(unittest.TestCase):
    def test_invokes_pip_with_the_pip_name_override(self):
        with patch("agent.dependencies.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = install("sklearn")
        self.assertTrue(result.ok)
        argv = run.call_args.args[0]
        self.assertIn("scikit-learn", argv)
        self.assertNotIn("sklearn", argv)

    def test_reports_a_nonzero_exit_as_failure_without_raising(self):
        with patch("agent.dependencies.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stdout="", stderr="ERROR: no matching distribution")
            result = install("definitely-not-a-real-package-xyz")
        self.assertFalse(result.ok)
        self.assertIn("failed", result.message)

    def test_refuses_an_unsafe_looking_package_name_without_touching_subprocess(self):
        with patch("agent.dependencies.subprocess.run") as run:
            result = install("os; rm -rf /")
        self.assertFalse(result.ok)
        run.assert_not_called()

    def test_never_raises_on_a_timeout(self):
        import subprocess as _subprocess
        with patch("agent.dependencies.subprocess.run", side_effect=_subprocess.TimeoutExpired("pip", 1)):
            result = install("some-slow-package")
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.message)


if __name__ == "__main__":
    unittest.main()
