from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from test_metal.config import ProjectConfig


class TestProjectConfigDefaults:
    def test_default_values(self):
        cfg = ProjectConfig()
        assert cfg.excel_header_row == 3
        assert cfg.excel_usecols == "B:CN"
        assert cfg.missing_threshold == 0.5
        assert cfg.r2_high == 0.8
        assert cfg.r2_medium == 0.6
        assert cfg.r2_min_feasible == 0.3
        assert cfg.slope_min_abs == 1e-10
        assert cfg.outputs_dir == Path("outputs")
        assert cfg.log_filename == "run.log"

    def test_custom_values(self):
        cfg = ProjectConfig(
            excel_header_row=5,
            excel_usecols="A:Z",
            missing_threshold=0.3,
            outputs_dir=Path("custom_outputs"),
        )
        assert cfg.excel_header_row == 5
        assert cfg.excel_usecols == "A:Z"
        assert cfg.missing_threshold == 0.3
        assert cfg.outputs_dir == Path("custom_outputs")

    def test_immutability(self):
        cfg = ProjectConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.excel_header_row = 5

    def test_frozen_dataclass(self):
        cfg = ProjectConfig()
        assert hasattr(cfg, "__dataclass_fields__")
        assert cfg.__dataclass_fields__["excel_header_row"].type is int
