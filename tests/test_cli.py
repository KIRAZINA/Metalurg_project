import subprocess
import sys


def _cli_command() -> list[str]:
    return [sys.executable, "-m", "test_metal"]


class TestCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [*_cli_command(), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--file" in result.stdout
        assert "--output" in result.stdout
        assert "--mode" in result.stdout

    def test_mode_choices_after_and_before(self):
        result = subprocess.run(
            [*_cli_command(), "--help"],
            capture_output=True,
            text=True,
        )
        assert "after" in result.stdout
        assert "before" in result.stdout

    def test_file_not_found_error_handled(self):
        result = subprocess.run(
            [*_cli_command(), "--file", "nonexistent_file.xls"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_entry_point_exists(self):
        result = subprocess.run(
            ["test-metal", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_version_not_defined(self):
        result = subprocess.run(
            [*_cli_command(), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_output_default_value(self):
        result = subprocess.run(
            [*_cli_command(), "--help"],
            capture_output=True,
            text=True,
        )
        assert "output" in result.stdout.lower()
