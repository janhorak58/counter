from __future__ import annotations

"""Compatibility layer.

Cíl: aby se sdílené I/O helpery používaly z jednoho místa.
Preferuj importy z `counter.core.io`, ale staré importy z `counter.core.utils` necháme fungovat.
"""

from counter.core.io.fs import ensure_dir

# Re-export for backward compatibility with older imports.

__all__ = ["ensure_dir"]
