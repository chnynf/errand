"""Command-line entrypoint for ``python -m errand``."""

import argparse
import asyncio

from errand.runtime.app import run_errand


def main() -> None:
    """Parse CLI args and start the unified Errand runtime."""
    parser = argparse.ArgumentParser(description="Run the Errand runtime.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--interface",
        action="append",
        choices=["discord", "cli", "web", "wechat"],
        help="Interface to start. Can be provided more than once. Defaults to config.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Start all known interfaces, regardless of config.json.",
    )
    args = parser.parse_args()

    interfaces = ["discord", "cli", "web", "wechat"] if args.all else args.interface
    try:
        asyncio.run(run_errand(debug=args.debug, interfaces=interfaces))
    except KeyboardInterrupt:
        print("\nErrand runtime stopped.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
