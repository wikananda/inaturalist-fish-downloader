"""Import guards for heavyweight vision dependencies."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def model_import_context():
    """Hide a repository-local Triton source checkout during model imports.

    A top-level ``triton/`` checkout is detected as a Python namespace package
    even when it is not installed. Torch then mistakes it for a working Triton
    runtime and lazy imports in Ultralytics, Transformers, torchvision, and SAM
    fail at ``triton.language``. Installed Triton packages outside this repository
    remain available.
    """
    project_root = Path(__file__).resolve().parents[2]
    local_triton = project_root / "triton"
    original_path = list(sys.path)
    if local_triton.is_dir():
        filtered_path = []
        for entry in sys.path:
            try:
                resolved = Path(entry or Path.cwd()).resolve()
            except (OSError, RuntimeError):
                resolved = None
            if resolved != project_root:
                filtered_path.append(entry)
        sys.path[:] = filtered_path
        loaded_triton = sys.modules.get("triton")
        module_paths = [
            Path(value).resolve()
            for value in getattr(loaded_triton, "__path__", [])
            if value
        ]
        if local_triton.resolve() in module_paths:
            sys.modules.pop("triton", None)
    try:
        yield
    finally:
        sys.path[:] = original_path
