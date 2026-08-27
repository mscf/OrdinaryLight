import subprocess
import sys
import textwrap
import unittest


class OptionalDependencyBoundaryTests(unittest.TestCase):
    def test_core_and_integration_modules_import_without_gui_packages(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            blocked = {"glfw", "PySide6", "dearpygui", "vulkan"}

            class BlockOptionalGuiPackages(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.partition(".")[0] in blocked:
                        raise ModuleNotFoundError(
                            f"optional GUI package blocked by test: {fullname}"
                        )
                    return None

            sys.meta_path.insert(0, BlockOptionalGuiPackages())

            import ordinarylight
            import ordinarylight.integrations.dearpygui
            import ordinarylight.integrations.glfw
            import ordinarylight.integrations.qt_workbench
            assert ordinarylight.backends.ReferenceBackend
            assert ordinarylight.RenderBackend
            from ordinarylight.validation import performance_gate_result

            result = performance_gate_result(
                60.0, 50.0, (3840, 2160), (3840, 2160), 0.98
            )
            assert result["status"] == "pass"
            """
        )
        completed = subprocess.run(
            (sys.executable, "-c", script),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
