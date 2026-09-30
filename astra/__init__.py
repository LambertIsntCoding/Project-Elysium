"""Astra: a Phase 1 local companion runtime.

Layers, lowest to highest:

* :mod:`astra.memory`   - persistent storage plus the pure memory-reliability
  rules (source hierarchy, strength, contradiction, governing selection,
  relationship boundaries).
* :mod:`astra.consolidator` - turns a conversation turn into candidate memories.
* :mod:`astra.orchestrator` - retrieval and prompt construction.
* :mod:`astra.elysium`  - the application-level root command interface.
* :mod:`main`           - the CLI that routes input between the layers.
"""
from __future__ import annotations

__version__ = "1.0.0"
