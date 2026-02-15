"""Entry point for Numerical Methods Visualizer."""

from __future__ import annotations

from ui import App


def main() -> None:
    """Application bootstrap."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
