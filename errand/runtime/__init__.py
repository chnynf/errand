"""Process lifecycle and component wiring.

This ``__init__`` intentionally does NOT re-export submodules. Lazy
imports keep leaf consumers (e.g. ``runtime.debug``) free of the
``ErrandApp`` import chain, which would otherwise force ``dotenv`` and
the rest of the runtime to load.

Import explicitly:

    from errand.runtime.app import ErrandApp, run_errand
    from errand.runtime.debug import debug_log, set_debug
"""
