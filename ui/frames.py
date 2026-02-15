"""Main UI frame implementations."""

from __future__ import annotations

import customtkinter as ctk

from .base import AppFrame
from .theme import FONTS, PALETTE


class LandingFrame(AppFrame):
    """Landing and onboarding screen."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        card = ctk.CTkFrame(
            self,
            fg_color=PALETTE["surface"],
            corner_radius=18,
            border_width=1,
            border_color="#223042",
        )
        card.grid(row=0, column=0, padx=48, pady=48, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        self._full_title = "Numerical Methods Visualizer"
        self._title_index = 0

        self.title_label = ctk.CTkLabel(
            card,
            text="",
            font=FONTS["title"],
            text_color=PALETTE["text_primary"],
        )
        self.title_label.grid(row=0, column=0, padx=24, pady=(28, 8), sticky="n")

        subtitle_text = (
            "Explore root-finding and linear-system iterative solvers with precise, "
            "iteration-by-iteration numerical feedback."
        )
        subtitle = ctk.CTkLabel(
            card,
            text=subtitle_text,
            justify="center",
            wraplength=860,
            font=FONTS["subtitle"],
            text_color=PALETTE["text_secondary"],
        )
        subtitle.grid(row=1, column=0, padx=28, pady=(0, 20))

        info_card = ctk.CTkFrame(card, fg_color=PALETTE["card"], corner_radius=14)
        info_card.grid(row=2, column=0, padx=28, pady=8, sticky="ew")
        info_card.grid_columnconfigure((0, 1, 2), weight=1)

        self._build_info_column(
            info_card,
            0,
            "Root-Finding",
            "Newton-Raphson and Regula Falsi locate x where f(x)=0.",
        )
        self._build_info_column(
            info_card,
            1,
            "Iterative Solvers",
            "Gauss-Jacobi and Gauss-Seidel solve Ax=b using repeated updates.",
        )
        self._build_info_column(
            info_card,
            2,
            "Convergence",
            "Track per-iteration error to verify stability and stopping criteria.",
        )

        cta = ctk.CTkButton(
            card,
            text="Get Started",
            width=220,
            height=44,
            corner_radius=10,
            font=FONTS["subtitle"],
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#04121E",
            command=lambda: controller.show_frame("SelectionFrame"),
        )
        cta.grid(row=3, column=0, pady=(24, 30))

    def _build_info_column(self, parent: ctk.CTkFrame, column: int, title: str, body: str) -> None:
        col_frame = ctk.CTkFrame(parent, fg_color="transparent")
        col_frame.grid(row=0, column=column, padx=14, pady=18, sticky="nsew")

        ctk.CTkLabel(
            col_frame,
            text=title,
            font=FONTS["heading"],
            text_color=PALETTE["text_primary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            col_frame,
            text=body,
            wraplength=250,
            justify="left",
            font=FONTS["body"],
            text_color=PALETTE["text_secondary"],
        ).pack(anchor="w", pady=(6, 0))

    def on_show(self) -> None:
        self._title_index = 0
        self.title_label.configure(text="")
        self._animate_title()

    def _animate_title(self) -> None:
        if self._title_index <= len(self._full_title):
            self.title_label.configure(text=self._full_title[: self._title_index])
            self._title_index += 1
            self.after(22, self._animate_title)


class SelectionFrame(AppFrame):
    """Placeholder for method selection screen."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])
        panel = ctk.CTkFrame(self, fg_color=PALETTE["surface"], corner_radius=16)
        panel.pack(expand=True, fill="both", padx=48, pady=48)

        ctk.CTkLabel(panel, text="Method Selection", font=FONTS["title"], text_color=PALETTE["text_primary"]).pack(pady=(40, 8))
        ctk.CTkLabel(
            panel,
            text="Cards and solver metadata will be added in the next phase.",
            font=FONTS["subtitle"],
            text_color=PALETTE["text_secondary"],
        ).pack()
        ctk.CTkButton(panel, text="Back", command=lambda: controller.show_frame("LandingFrame")).pack(pady=24)


class InputFrame(AppFrame):
    """Placeholder for dynamic input forms."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])


class ResultFrame(AppFrame):
    """Placeholder for result rendering."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])
