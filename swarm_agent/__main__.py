"""Command-line entrypoint for ``python -m swarm_agent``."""

import argparse
import asyncio

from swarm_agent.runtime.app import run_swarm


def main() -> None:
    """Parse CLI args and start the unified Swarm runtime."""
    parser = argparse.ArgumentParser(description="Run the Swarm runtime.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--interface",
        action="append",
        choices=["discord", "cli", "web"],
        help="Interface to start. Can be provided more than once. Defaults to config.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Start all known interfaces, regardless of config.json.",
    )
    args = parser.parse_args()

    interfaces = ["discord", "cli", "web"] if args.all else args.interface
    try:
        asyncio.run(run_swarm(debug=args.debug, interfaces=interfaces))
    except KeyboardInterrupt:
        print("\nSwarm runtime stopped.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
