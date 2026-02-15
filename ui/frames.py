"""Main UI frame implementations."""

from __future__ import annotations

from dataclasses import dataclass

import customtkinter as ctk

from .base import AppFrame
from .theme import FONTS, PALETTE


@dataclass(slots=True)
class MethodCardSpec:
    """Metadata used to render each selection card."""

    key: str
    name: str
    description: str
    formula: str


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
    """Card-based method selection screen."""

    CARD_COLOR = "#1A212D"
    CARD_HOVER_COLOR = "#233044"

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        panel = ctk.CTkFrame(
            self,
            fg_color=PALETTE["surface"],
            corner_radius=16,
            border_width=1,
            border_color="#26364C",
        )
        panel.grid(row=0, column=0, padx=42, pady=42, sticky="nsew")
        panel.grid_columnconfigure((0, 1), weight=1)
        panel.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            panel,
            text="Choose a Numerical Method",
            font=FONTS["title"],
            text_color=PALETTE["text_primary"],
        ).grid(row=0, column=0, columnspan=2, pady=(26, 6))
        ctk.CTkLabel(
            panel,
            text="Select a strategy to configure inputs and run iteration analysis.",
            font=FONTS["subtitle"],
            text_color=PALETTE["text_secondary"],
        ).grid(row=1, column=0, columnspan=2, pady=(0, 20))

        self.method_specs = [
            MethodCardSpec(
                key="newton_raphson",
                name="Newton-Raphson",
                description="Fast tangent-based root refinement from one initial guess.",
                formula="x(n+1) = x(n) - f(x(n)) / f'(x(n))",
            ),
            MethodCardSpec(
                key="regula_falsi",
                name="Regula Falsi",
                description="Bracketed false-position method using interval sign changes.",
                formula="c = (a f(b) - b f(a)) / (f(b) - f(a))",
            ),
            MethodCardSpec(
                key="gauss_jacobi",
                name="Gauss-Jacobi",
                description="Parallel-friendly iterative solver using prior vector values.",
                formula="x_i(k+1) = (b_i - Σ(j!=i) a_ij x_j(k)) / a_ii",
            ),
            MethodCardSpec(
                key="gauss_seidel",
                name="Gauss-Seidel",
                description="Sequential iterative solver with immediate in-step updates.",
                formula="x_i(k+1) = (b_i - Σ(j<i) a_ij x_j(k+1) - Σ(j>i) a_ij x_j(k)) / a_ii",
            ),
        ]

        cards_frame = ctk.CTkFrame(panel, fg_color="transparent")
        cards_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=24, pady=10)
        cards_frame.grid_columnconfigure((0, 1), weight=1)
        cards_frame.grid_rowconfigure((0, 1), weight=1)

        for index, spec in enumerate(self.method_specs):
            row = index // 2
            col = index % 2
            card = self._build_method_card(cards_frame, spec)
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")

        footer = ctk.CTkFrame(panel, fg_color="transparent")
        footer.grid(row=3, column=0, columnspan=2, pady=(6, 20))
        ctk.CTkButton(
            footer,
            text="Back",
            width=120,
            fg_color="#29364A",
            hover_color="#374A66",
            command=lambda: controller.show_frame("LandingFrame"),
        ).pack()

    def _build_method_card(self, parent: ctk.CTkFrame, spec: MethodCardSpec) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=self.CARD_COLOR,
            corner_radius=14,
            border_width=1,
            border_color="#2A3A50",
        )

        title = ctk.CTkLabel(card, text=spec.name, font=FONTS["heading"], text_color=PALETTE["text_primary"])
        title.pack(anchor="w", padx=16, pady=(14, 6))

        description = ctk.CTkLabel(
            card,
            text=spec.description,
            wraplength=420,
            justify="left",
            font=FONTS["body"],
            text_color=PALETTE["text_secondary"],
        )
        description.pack(anchor="w", padx=16)

        formula = ctk.CTkLabel(
            card,
            text=spec.formula,
            wraplength=420,
            justify="left",
            font=FONTS["mono"],
            text_color="#8DD5FF",
        )
        formula.pack(anchor="w", padx=16, pady=(10, 12))

        select_button = ctk.CTkButton(
            card,
            text="Select",
            width=100,
            height=34,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#03111D",
            command=lambda selected=spec.key: self._select_method(selected),
        )
        select_button.pack(anchor="w", padx=16, pady=(0, 14))

        self._bind_card_hover(card, [title, description, formula, select_button])
        return card

    def _bind_card_hover(self, card: ctk.CTkFrame, children: list[ctk.CTkBaseClass]) -> None:
        targets = [card, *children]
        for target in targets:
            target.bind("<Enter>", lambda _event, c=card: c.configure(fg_color=self.CARD_HOVER_COLOR))
            target.bind("<Leave>", lambda _event, c=card: c.configure(fg_color=self.CARD_COLOR))

    def _select_method(self, method_key: str) -> None:
        self.controller.selected_method = method_key
        self.controller.show_frame("InputFrame")


class InputFrame(AppFrame):
    """Placeholder for dynamic input forms."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])


class ResultFrame(AppFrame):
    """Placeholder for result rendering."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])
