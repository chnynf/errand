"""Shared pytest configuration.

Marks the whole `tests/` tree as asyncio-driven so individual test
functions don't need a per-test `@pytest.mark.asyncio` decorator.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import swarm_agent` works
# whether pytest is invoked from the repo root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
