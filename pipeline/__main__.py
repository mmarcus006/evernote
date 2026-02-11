"""Allow ``python -m pipeline`` to run the CLI."""

from .cli import main

raise SystemExit(main())
