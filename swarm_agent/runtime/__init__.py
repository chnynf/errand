"""Process lifecycle and component wiring.

This ``__init__`` intentionally does NOT re-export submodules. Lazy
imports keep leaf consumers (e.g. ``runtime.debug``) free of the
``SwarmApp`` import chain, which would otherwise force ``dotenv`` and
the rest of the runtime to load.

Import explicitly:

    from swarm_agent.runtime.app import SwarmApp, run_swarm
    from swarm_agent.runtime.debug import debug_log, set_debug
"""
