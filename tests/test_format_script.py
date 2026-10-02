"""Tests for scripts/format.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import format as fmt_script


def test_format_script_main_check_success():
    with patch("format.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        ret = fmt_script.main(["--check", "src"])
        assert ret == 0
        assert mock_run.call_count == 2  # format --check and check


def test_format_script_main_format_success():
    with patch("format.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        ret = fmt_script.main(["src"])
        assert ret == 0
        assert mock_run.call_count == 2  # format and check --fix
