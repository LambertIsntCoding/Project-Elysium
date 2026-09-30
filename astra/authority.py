"""Backwards-compatibility shim: authority logic now lives in ``memory.py``.

Importing ``astra.authority`` (or the legacy top-level ``memory_authority``)
still works; it simply re-exports from :mod:`astra.memory`.
"""
from .memory import *  # noqa: F401,F403
from .memory import (  # noqa: F401
    ARCHIVE_CONFIDENCE_THRESHOLD,
    ARCHIVE_MIN_AGE_DAYS,
    ARCHIVE_STRENGTH_THRESHOLD,
    ARCHIVE_UNUSED_DAYS,
    GOVERNING_PRIORITY,
    GOVERNING_TYPES,
    MAX_GOVERNING_MEMORIES,
    RELATIONSHIP_BOUNDARIES,
    SOURCE_TIERS,
    apply_relationship_guard,
    boundary_memories,
    detect_contradiction,
    effective_strength,
    governing_priority,
    half_life_days,
    is_astra_source,
    is_governing,
    is_governing_eligible,
    memory_importance,
    recency_factor,
    relationship_risk,
    select_governing,
    source_reliability,
    source_tier,
)
