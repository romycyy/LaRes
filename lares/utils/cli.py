"""Small helpers every ``scripts/*.py`` entry point used to carry its own copy of."""

from __future__ import annotations

import os

import yaml


def separator(title: str, width: int = 80) -> None:
    rule = "=" * width
    print(f"\n{rule}\n  {title}\n{rule}")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_api_key(env_var: str = "OPENAI_API_KEY", min_length: int = 10) -> str:
    """The LLM API key, or a clean exit naming the variable that is missing."""
    key = os.environ.get(env_var, "")
    if len(key) < min_length:
        raise SystemExit(f"{env_var} is not set")
    return key


def openai_client(**kwargs):
    """An OpenAI client built from :func:`require_api_key`; imports lazily so ledger CLIs stay light."""
    from openai import OpenAI

    return OpenAI(api_key=require_api_key(), **kwargs)


def self_argv(config_path: str, script: str | None = None) -> list[str]:
    """The argv prefix that re-runs the current entry point as a child: interpreter, ``-u``, script, ``--config``.

    Every one-rollout-per-process protocol spawns the script it is running in; this is the one spelling of that
    prefix. ``script`` defaults to the running script (``sys.argv[0]``) as an absolute path.
    """
    import sys

    return [sys.executable, "-u", os.path.abspath(script or sys.argv[0]), "--config", config_path]
