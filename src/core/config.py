"""Configuration utilities for FP-DETR."""

import yaml
from argparse import Namespace


def load_config(path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place) and return result."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            merge_config(base[k], v)
        else:
            base[k] = v
    return base


def dict_to_namespace(d: dict) -> Namespace:
    """Convert a (possibly nested) dict to argparse.Namespace."""
    ns = Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns
