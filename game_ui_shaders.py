"""Load GLSL sources from disk."""

from __future__ import annotations


def load_shader(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()
