"""LaRes utilities.

``env_wrapper`` and ``make_metaworld_env`` are resolved lazily so that the
pure-Python helpers here (hashing, git state, seeds, env ids) import without
MetaWorld or MuJoCo installed.
"""

__all__ = ["env_wrapper", "make_metaworld_env"]


def __getattr__(name):
    if name in __all__:
        from lares.utils import metaworld_env

        return getattr(metaworld_env, name)
    raise AttributeError(f"module 'lares.utils' has no attribute {name!r}")
