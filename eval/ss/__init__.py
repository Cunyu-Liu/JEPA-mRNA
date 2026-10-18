"""Evaluation, ablation and downstream-task pipeline for the RNA SS decision model.

This package implements spec §6 (downstream tasks), §7 (metrics / statistics /
ablations) and §8 (quantitative gates).  It is a *research* package: the modules
are deliberately small and focused, and every module that would need an external
tool that is not installed here (ViennaRNA / RNAstructure / LinearPartition /
MMseqs2) ships a clean stub that raises instead of silently faking output.

Import convention
-----------------
The package lives outside ``src/``.  ``eval/`` must be on ``sys.path`` (the test
suite adds it) so that ``import ss`` works; this ``__init__`` also puts the
project's ``src/`` on ``sys.path`` so the frozen ``rnajepa`` modules resolve.
``python -m ss.gates`` works when run from the ``eval/`` directory.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

__all__ = ["_ROOT", "_SRC"]
