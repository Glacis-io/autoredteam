"""Allow ``python -m autoredteam`` to use the packaged CLI."""

from cli import main


if __name__ == "__main__":
    raise SystemExit(main())
