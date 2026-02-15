"""Base frame class used across all application screens."""

from __future__ import annotations

import customtkinter as ctk


class AppFrame(ctk.CTkFrame):
    """Base frame with lifecycle hooks."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.controller = controller

    def on_show(self) -> None:
        """Called whenever the frame is raised."""
        return
