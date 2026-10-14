"""Application entry point for the AI Watermark Remover desktop tool."""
from __future__ import annotations

import sys


def main() -> int:
    # Lazy import so that ``--help`` style CLI use does not require a display.
    from PySide6.QtWidgets import QApplication

    from src.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AI Watermark Remover")
    app.setOrganizationName("WatermarkRemoval")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
