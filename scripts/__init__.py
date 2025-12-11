"""Helper utilities for CLI entry points.

This makes the ``scripts`` directory importable in unit tests that rely on
helpers such as ``create_synthetic_data``.
"""

# The CLI modules are intentionally re-exported on demand. Tests import
# specific modules (e.g., scripts.create_synthetic_data) so no explicit
# exports are required here.
