"""Module entrypoint allowing `python -m attentionvision <subcommand>`."""

from __future__ import annotations

import sys

from attentionvision.cli import main

if __name__ == "__main__":
    sys.exit(main())
