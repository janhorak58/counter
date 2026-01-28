from __future__ import annotations

"""Compatibility layer for legacy imports.

Prefer imports from ``counter.core.io``. This module keeps older
``counter.core.utils`` imports working without duplication.
"""

from counter.core.io.fs import ensure_dir

__all__ = ["ensure_dir"]
