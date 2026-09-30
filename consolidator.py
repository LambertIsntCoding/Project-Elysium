"""Compatibility shim: the module now lives in :mod:`astra.consolidator`.

Re-exports the public API and forwards any other attribute (including
private helpers) so existing imports keep working unchanged.
"""
import astra.consolidator as _mod
from astra.consolidator import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_mod, name)


def __dir__():
    return sorted(set(dir(_mod)) | set(globals()))
