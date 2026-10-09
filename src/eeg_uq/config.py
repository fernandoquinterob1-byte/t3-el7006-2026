"""Single, validated source for experiment defaults."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_project_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the one project configuration used by SEED and Conformal CLIs."""

    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Project configuration must be a YAML mapping.")
    for section in ("training", "conformal", "paths", "laplace", "uncertainty"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"Project configuration requires a {section!r} section.")
    return config


def configured(cli_value: Any, section: Mapping[str, Any], key: str) -> Any:
    """Use an explicit CLI override; otherwise use the shared YAML value."""

    if cli_value is not None:
        return cli_value
    if key not in section:
        raise ValueError(f"Configuration is missing {key!r}.")
    return section[key]
