"""Configuration loading and validation utilities."""

import os
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads and validates a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing or invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Basic validation of mandatory sections
    required_sections = ["vision", "language", "projector", "training", "data"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}' in {config_path}")

    return config
