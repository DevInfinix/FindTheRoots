"""Main UI frame implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import customtkinter as ctk

from .base import AppFrame
from .strategies import MethodStrategyFactory, SolveRequest
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
    """Dynamic method-specific input form screen."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])
        self.method_names = {
            "newton_raphson": "Newton-Raphson",
            "regula_falsi": "Regula Falsi",
            "gauss_jacobi": "Gauss-Jacobi",
            "gauss_seidel": "Gauss-Seidel",
        }

        self.matrix_size = ctk.StringVar(value="3")
        self.function_entry: ctk.CTkEntry | None = None
        self.initial_guess_entry: ctk.CTkEntry | None = None
        self.lower_bound_entry: ctk.CTkEntry | None = None
        self.upper_bound_entry: ctk.CTkEntry | None = None
        self.iterations_entry: ctk.CTkEntry | None = None
        self.precision_entry: ctk.CTkEntry | None = None
        self.tolerance_entry: ctk.CTkEntry | None = None
        self.matrix_entries: list[list[ctk.CTkEntry]] = []
        self.constants_entries: list[ctk.CTkEntry] = []
        self.initial_vector_entries: list[ctk.CTkEntry] = []

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.container = ctk.CTkFrame(
            self,
            fg_color=PALETTE["surface"],
            corner_radius=16,
            border_width=1,
            border_color="#26364C",
        )
        self.container.grid(row=0, column=0, padx=38, pady=38, sticky="nsew")
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.container,
            text="Configure Method",
            font=FONTS["title"],
            text_color=PALETTE["text_primary"],
        )
        self.title_label.grid(row=0, column=0, pady=(20, 6))

        self.form_scroll = ctk.CTkScrollableFrame(self.container, fg_color=PALETTE["card"], corner_radius=14)
        self.form_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(6, 12))
        self.form_scroll.grid_columnconfigure(0, weight=1)

        self.error_label = ctk.CTkLabel(
            self.container,
            text="",
            font=FONTS["body"],
            text_color=PALETTE["danger"],
        )
        self.error_label.grid(row=2, column=0, pady=(0, 6))

        footer = ctk.CTkFrame(self.container, fg_color="transparent")
        footer.grid(row=3, column=0, pady=(0, 20))
        ctk.CTkButton(
            footer,
            text="Back",
            width=110,
            fg_color="#29364A",
            hover_color="#374A66",
            command=lambda: controller.show_frame("SelectionFrame"),
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            footer,
            text="Solve",
            width=150,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#04121E",
            command=self._solve_current_method,
        ).pack(side="left", padx=8)

    def on_show(self) -> None:
        method_key = self.controller.selected_method or "newton_raphson"
        method_name = self.method_names.get(method_key, "Method")
        self.title_label.configure(text=f"Configure {method_name}")
        self.error_label.configure(text="")
        self._render_method_form(method_key)

    def _render_method_form(self, method_key: str) -> None:
        for child in self.form_scroll.winfo_children():
            child.destroy()

        self.function_entry = None
        self.initial_guess_entry = None
        self.lower_bound_entry = None
        self.upper_bound_entry = None
        self.matrix_entries = []
        self.constants_entries = []
        self.initial_vector_entries = []

        row = 0
        if method_key in {"newton_raphson", "regula_falsi"}:
            self.function_entry = self._add_entry_row(row, "f(x)", "x**3 - x - 2")
            row += 1

            if method_key == "newton_raphson":
                self.initial_guess_entry = self._add_entry_row(row, "Initial Guess", "1.5")
            else:
                self.lower_bound_entry = self._add_entry_row(row, "Lower Bound (a)", "1")
                row += 1
                self.upper_bound_entry = self._add_entry_row(row, "Upper Bound (b)", "2")
            row += 1
        else:
            self._add_matrix_size_selector(row)
            row += 1
            self._build_matrix_grid(size=int(self.matrix_size.get()), row=row)
            row += 1

        self.iterations_entry = self._add_entry_row(row, "Max Iterations", "50")
        row += 1
        self.precision_entry = self._add_entry_row(row, "Decimal Precision", "6")
        row += 1
        self.tolerance_entry = self._add_entry_row(row, "Tolerance", "1e-8")

    def _add_entry_row(self, row: int, label: str, placeholder: str) -> ctk.CTkEntry:
        wrapper = ctk.CTkFrame(self.form_scroll, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky="ew", padx=16, pady=8)
        wrapper.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(wrapper, text=label, width=180, anchor="w", font=FONTS["body"]).grid(row=0, column=0, padx=(0, 8))
        entry = ctk.CTkEntry(wrapper, placeholder_text=placeholder, fg_color="#111824", border_color="#32445E")
        entry.grid(row=0, column=1, sticky="ew")
        return entry

    def _add_matrix_size_selector(self, row: int) -> None:
        wrapper = ctk.CTkFrame(self.form_scroll, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky="ew", padx=16, pady=8)
        wrapper.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(wrapper, text="Matrix Size", width=180, anchor="w", font=FONTS["body"]).grid(row=0, column=0, padx=(0, 8))
        selector = ctk.CTkOptionMenu(
            wrapper,
            variable=self.matrix_size,
            values=["2", "3", "4", "5", "6"],
            fg_color="#2A3A50",
            button_color="#324760",
            button_hover_color="#3E5676",
            command=lambda value: self._build_matrix_grid(size=int(value), row=row + 1),
        )
        selector.grid(row=0, column=1, sticky="w")

    def _build_matrix_grid(self, size: int, row: int) -> None:
        current = self.form_scroll.grid_slaves(row=row, column=0)
        for widget in current:
            widget.destroy()

        panel = ctk.CTkFrame(self.form_scroll, fg_color="#111824", corner_radius=12)
        panel.grid(row=row, column=0, sticky="ew", padx=16, pady=8)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text="Enter A matrix, b constants, and initial guess vector x(0)",
            font=FONTS["body"],
            text_color=PALETTE["text_secondary"],
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 8))

        grid_frame = ctk.CTkFrame(panel, fg_color="transparent")
        grid_frame.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="w")

        self.matrix_entries = []
        self.constants_entries = []
        self.initial_vector_entries = []

        for i in range(size):
            matrix_row_entries: list[ctk.CTkEntry] = []
            for j in range(size):
                entry = ctk.CTkEntry(grid_frame, width=68, placeholder_text="0")
                entry.grid(row=i, column=j, padx=4, pady=4)
                entry.insert(0, "0")
                matrix_row_entries.append(entry)
            ctk.CTkLabel(grid_frame, text="|", font=FONTS["heading"]).grid(row=i, column=size, padx=8)
            b_entry = ctk.CTkEntry(grid_frame, width=68, placeholder_text="0")
            b_entry.grid(row=i, column=size + 1, padx=4, pady=4)
            b_entry.insert(0, "0")
            self.constants_entries.append(b_entry)

            ctk.CTkLabel(grid_frame, text="x0", font=FONTS["body"]).grid(row=i, column=size + 2, padx=(12, 4))
            x0_entry = ctk.CTkEntry(grid_frame, width=68, placeholder_text="0")
            x0_entry.grid(row=i, column=size + 3, padx=4, pady=4)
            x0_entry.insert(0, "0")
            self.initial_vector_entries.append(x0_entry)

            self.matrix_entries.append(matrix_row_entries)

    def _parse_float(self, entry: ctk.CTkEntry, field_name: str) -> float:
        value = entry.get().strip()
        if not value:
            raise ValueError(f"{field_name} is required.")
        return float(value)

    def _parse_int(self, entry: ctk.CTkEntry, field_name: str, minimum: int = 1) -> int:
        value = entry.get().strip()
        if not value:
            raise ValueError(f"{field_name} is required.")
        parsed = int(value)
        if parsed < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}.")
        return parsed

    def _collect_payload(self, method_key: str) -> dict[str, Any]:
        max_iterations = self._parse_int(self.iterations_entry, "Max Iterations", minimum=1)
        precision = self._parse_int(self.precision_entry, "Decimal Precision", minimum=0)
        tolerance = self._parse_float(self.tolerance_entry, "Tolerance")
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive.")

        payload: dict[str, Any] = {
            "max_iterations": max_iterations,
            "precision": precision,
            "tolerance": tolerance,
        }

        if method_key in {"newton_raphson", "regula_falsi"}:
            function_expression = (self.function_entry.get() if self.function_entry else "").strip()
            if not function_expression:
                raise ValueError("Function expression is required.")
            payload["function_expression"] = function_expression

            if method_key == "newton_raphson":
                payload["initial_guess"] = self._parse_float(self.initial_guess_entry, "Initial Guess")
            else:
                payload["lower_bound"] = self._parse_float(self.lower_bound_entry, "Lower Bound")
                payload["upper_bound"] = self._parse_float(self.upper_bound_entry, "Upper Bound")
        else:
            if not self.matrix_entries:
                raise ValueError("Matrix entries are not initialized.")
            matrix = [
                [self._parse_float(entry, f"A[{row + 1},{col + 1}]") for col, entry in enumerate(row_entries)]
                for row, row_entries in enumerate(self.matrix_entries)
            ]
            constants = [self._parse_float(entry, f"b[{idx + 1}]") for idx, entry in enumerate(self.constants_entries)]
            initial_guess = [
                self._parse_float(entry, f"x0[{idx + 1}]") for idx, entry in enumerate(self.initial_vector_entries)
            ]
            payload["matrix"] = matrix
            payload["constants"] = constants
            payload["initial_guess"] = initial_guess

        return payload

    def _solve_current_method(self) -> None:
        method_key = self.controller.selected_method
        if not method_key:
            self.error_label.configure(text="No method selected. Go back and choose a method first.")
            return

        try:
            payload = self._collect_payload(method_key)
            request = SolveRequest(method_key=method_key, data=payload)
            solver = MethodStrategyFactory.create_solver(request)
            result = solver.solve()
        except Exception as exc:
            self.error_label.configure(text=str(exc))
            return

        self.error_label.configure(text="")
        self.controller.last_result = result
        self.controller.last_request = payload
        self.controller.show_frame("ResultFrame")


class ResultFrame(AppFrame):
    """Placeholder for result rendering."""

    def __init__(self, parent: ctk.CTkFrame, controller: ctk.CTk) -> None:
        super().__init__(parent, controller, fg_color=PALETTE["bg"])
