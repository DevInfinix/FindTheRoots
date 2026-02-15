import random

import customtkinter as ctk
import numpy as np

from .effects import ConfettiLayer
from .strategies import create_solver
from .theme import FONTS, PALETTE


class LandingFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=PALETTE["bg"])
        self.controller = controller
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        card = ctk.CTkFrame(self, fg_color=PALETTE["surface"], corner_radius=20)
        card.grid(row=0, column=0, padx=50, pady=50, sticky="nsew")

        self.full_title = "Numerical Methods Visualizer"
        self.title_index = 0

        self.title_label = ctk.CTkLabel(card, text="", font=FONTS["title"], text_color=PALETTE["text_primary"])
        self.title_label.pack(pady=(40, 12))

        ctk.CTkLabel(
            card,
            text="Newton-Raphson, Regula Falsi, Gauss-Jacobi, and Gauss-Seidel in a simple interface.",
            font=FONTS["subtitle"],
            text_color=PALETTE["text_secondary"],
            wraplength=820,
            justify="center",
        ).pack(pady=(0, 20))

        facts = ctk.CTkFrame(card, fg_color=PALETTE["card"], corner_radius=14)
        facts.pack(fill="x", padx=32, pady=(0, 24))

        for text in [
            "Root methods solve f(x)=0.",
            "Iterative solvers handle Ax=b.",
            "See each step in an iteration table.",
        ]:
            ctk.CTkLabel(facts, text="- " + text, anchor="w", font=FONTS["body"], text_color=PALETTE["text_secondary"]).pack(fill="x", padx=16, pady=6)

        ctk.CTkButton(
            card,
            text="Get Started",
            width=220,
            height=44,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#04121E",
            font=FONTS["body_bold"],
            command=lambda: controller.show_frame("SelectionFrame"),
        ).pack(pady=(0, 34))

    def on_show(self):
        self.title_index = 0
        self.title_label.configure(text="")
        self.animate_title()

    def animate_title(self):
        if self.title_index <= len(self.full_title):
            self.title_label.configure(text=self.full_title[: self.title_index])
            self.title_index += 1
            self.after(25, self.animate_title)


class SelectionFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=PALETTE["bg"])
        self.controller = controller
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        panel = ctk.CTkFrame(self, fg_color=PALETTE["surface"], corner_radius=20)
        panel.grid(row=0, column=0, padx=40, pady=40, sticky="nsew")
        panel.grid_columnconfigure((0, 1), weight=1)
        panel.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(panel, text="Choose a Method", font=FONTS["title"], text_color=PALETTE["text_primary"]).grid(row=0, column=0, columnspan=2, pady=(24, 8))
        ctk.CTkLabel(panel, text="Pick one card to continue.", font=FONTS["subtitle"], text_color=PALETTE["text_secondary"]).grid(row=1, column=0, columnspan=2, pady=(0, 14))

        self.methods = [
            {
                "key": "newton_raphson",
                "name": "Newton-Raphson",
                "description": "Start with one guess and improve it using slope each iteration.",
            },
            {
                "key": "regula_falsi",
                "name": "Regula Falsi",
                "description": "Use an interval and shrink it where sign changes.",
            },
            {
                "key": "gauss_jacobi",
                "name": "Gauss-Jacobi",
                "description": "Use only previous-iteration values to compute the next vector.",
            },
            {
                "key": "gauss_seidel",
                "name": "Gauss-Seidel",
                "description": "Use freshly updated values immediately in the same iteration.",
            },
        ]

        cards = ctk.CTkFrame(panel, fg_color="transparent")
        cards.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=22, pady=10)
        cards.grid_columnconfigure((0, 1), weight=1)
        cards.grid_rowconfigure((0, 1), weight=1)

        for index, method in enumerate(self.methods):
            card = self.build_card(cards, method)
            card.grid(row=index // 2, column=index % 2, padx=10, pady=10, sticky="nsew")

        ctk.CTkButton(
            panel,
            text="Back",
            width=120,
            fg_color="#2C3C54",
            hover_color="#3A4D68",
            command=lambda: controller.show_frame("LandingFrame"),
        ).grid(row=3, column=0, columnspan=2, pady=(8, 22))

    def build_card(self, parent, method):
        card = ctk.CTkFrame(parent, fg_color="#1A2434", corner_radius=14)
        ctk.CTkLabel(card, text=method["name"], font=FONTS["heading"], text_color=PALETTE["text_primary"]).pack(anchor="w", padx=14, pady=(14, 6))
        ctk.CTkLabel(card, text=method["description"], wraplength=420, justify="left", font=FONTS["body"], text_color=PALETTE["text_secondary"]).pack(anchor="w", padx=14, pady=(0, 10))

        button = ctk.CTkButton(
            card,
            text="Select",
            width=100,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_hover"],
            text_color="#04121E",
            command=lambda key=method["key"]: self.select_method(key),
        )
        button.pack(anchor="w", padx=14, pady=(0, 12))

        for widget in (card, button):
            widget.bind("<Enter>", lambda _event, c=card: c.configure(fg_color="#25344A"))
            widget.bind("<Leave>", lambda _event, c=card: c.configure(fg_color="#1A2434"))
        return card

    def select_method(self, key):
        self.controller.selected_method = key
        self.controller.show_frame("InputFrame")


class InputFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=PALETTE["bg"])
        self.controller = controller
        self.method_names = {
            "newton_raphson": "Newton-Raphson",
            "regula_falsi": "Regula Falsi",
            "gauss_jacobi": "Gauss-Jacobi",
            "gauss_seidel": "Gauss-Seidel",
        }

        self.matrix_size = ctk.StringVar(value="3")
        self.function_entry = None
        self.initial_guess_entry = None
        self.lower_bound_entry = None
        self.upper_bound_entry = None
        self.iterations_entry = None
        self.precision_entry = None
        self.matrix_entries = []
        self.constants_entries = []

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.container = ctk.CTkFrame(self, fg_color=PALETTE["surface"], corner_radius=18)
        self.container.grid(row=0, column=0, padx=38, pady=38, sticky="nsew")
        self.container.grid_rowconfigure(1, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(self.container, text="Configure Method", font=FONTS["title"], text_color=PALETTE["text_primary"])
        self.title_label.grid(row=0, column=0, pady=(18, 8))

        self.form_scroll = ctk.CTkScrollableFrame(self.container, fg_color=PALETTE["card"], corner_radius=14)
        self.form_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 12))
        self.form_scroll.grid_columnconfigure(0, weight=1)

        self.error_label = ctk.CTkLabel(self.container, text="", font=FONTS["body"], text_color=PALETTE["danger"])
        self.error_label.grid(row=2, column=0, pady=(0, 6))

        footer = ctk.CTkFrame(self.container, fg_color="transparent")
        footer.grid(row=3, column=0, pady=(0, 18))

        ctk.CTkButton(footer, text="Back", width=100, fg_color="#2C3C54", hover_color="#3A4D68", command=lambda: controller.show_frame("SelectionFrame")).pack(side="left", padx=6)
        ctk.CTkButton(footer, text="\U0001F3B2 Random", width=120, fg_color="#26653A", hover_color="#2F7E46", command=self.fill_random).pack(side="left", padx=6)
        ctk.CTkButton(footer, text="Solve", width=130, fg_color=PALETTE["accent"], hover_color=PALETTE["accent_hover"], text_color="#04121E", font=FONTS["body_bold"], command=self.solve_current_method).pack(side="left", padx=6)

    def on_show(self):
        method = self.controller.selected_method or "newton_raphson"
        self.title_label.configure(text="Configure " + self.method_names[method])
        self.error_label.configure(text="")
        self.render_form(method)

    def render_form(self, method):
        for child in self.form_scroll.winfo_children():
            child.destroy()

        self.function_entry = None
        self.initial_guess_entry = None
        self.lower_bound_entry = None
        self.upper_bound_entry = None
        self.matrix_entries = []
        self.constants_entries = []

        row = 0
        if method in ("newton_raphson", "regula_falsi"):
            self.function_entry = self.add_entry(row, "f(x)", "x**3 - x - 2")
            row += 1
            if method == "newton_raphson":
                self.initial_guess_entry = self.add_entry(row, "Initial Guess", "1.5")
                row += 1
            else:
                self.lower_bound_entry = self.add_entry(row, "Lower Bound (a)", "1")
                row += 1
                self.upper_bound_entry = self.add_entry(row, "Upper Bound (b)", "2")
                row += 1
        else:
            self.add_matrix_size(row)
            row += 1
            self.build_matrix_grid(int(self.matrix_size.get()), row)
            row += 1

        self.iterations_entry = self.add_entry(row, "Iterations", "20")
        row += 1
        self.precision_entry = self.add_entry(row, "Decimal Precision", "6")

    def add_entry(self, row, label, placeholder):
        wrap = ctk.CTkFrame(self.form_scroll, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=16, pady=7)
        wrap.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(wrap, text=label, width=170, anchor="w", font=FONTS["body"]).grid(row=0, column=0, padx=(0, 8))
        entry = ctk.CTkEntry(wrap, placeholder_text=placeholder, fg_color="#101824", border_color="#32445E")
        entry.grid(row=0, column=1, sticky="ew")
        return entry

    def add_matrix_size(self, row):
        wrap = ctk.CTkFrame(self.form_scroll, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=16, pady=7)
        ctk.CTkLabel(wrap, text="Matrix Size", width=170, anchor="w", font=FONTS["body"]).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkOptionMenu(wrap, variable=self.matrix_size, values=["2", "3", "4", "5"], command=lambda value: self.build_matrix_grid(int(value), row + 1)).grid(row=0, column=1, sticky="w")

    def build_matrix_grid(self, size, row):
        for widget in self.form_scroll.grid_slaves(row=row, column=0):
            widget.destroy()

        panel = ctk.CTkFrame(self.form_scroll, fg_color="#101824", corner_radius=12)
        panel.grid(row=row, column=0, sticky="ew", padx=16, pady=6)

        ctk.CTkLabel(panel, text="Enter matrix A and vector b. x(0) is zero vector.", font=FONTS["body"], text_color=PALETTE["text_secondary"]).grid(row=0, column=0, padx=12, pady=(10, 8), sticky="w")
        grid = ctk.CTkFrame(panel, fg_color="transparent")
        grid.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        self.matrix_entries = []
        self.constants_entries = []

        for i in range(size):
            row_entries = []
            for j in range(size):
                entry = ctk.CTkEntry(grid, width=62)
                entry.grid(row=i, column=j, padx=3, pady=3)
                entry.insert(0, "0")
                row_entries.append(entry)
            ctk.CTkLabel(grid, text="|", font=FONTS["body_bold"]).grid(row=i, column=size, padx=8)
            b_entry = ctk.CTkEntry(grid, width=62)
            b_entry.grid(row=i, column=size + 1, padx=3, pady=3)
            b_entry.insert(0, "0")
            self.constants_entries.append(b_entry)
            self.matrix_entries.append(row_entries)

    def parse_float(self, entry, label):
        text = entry.get().strip()
        if not text:
            raise ValueError(label + " is required.")
        return float(text)

    def parse_int(self, entry, label, minimum):
        text = entry.get().strip()
        if not text:
            raise ValueError(label + " is required.")
        value = int(text)
        if value < minimum:
            raise ValueError(label + " should be at least " + str(minimum) + ".")
        return value

    def collect_payload(self, method):
        payload = {
            "iterations": self.parse_int(self.iterations_entry, "Iterations", 1),
            "precision": self.parse_int(self.precision_entry, "Decimal Precision", 0),
        }

        if method in ("newton_raphson", "regula_falsi"):
            function_text = self.function_entry.get().strip() if self.function_entry else ""
            if not function_text:
                raise ValueError("f(x) is required.")
            payload["function_expression"] = function_text
            if method == "newton_raphson":
                payload["initial_guess"] = self.parse_float(self.initial_guess_entry, "Initial Guess")
            else:
                payload["lower_bound"] = self.parse_float(self.lower_bound_entry, "Lower Bound")
                payload["upper_bound"] = self.parse_float(self.upper_bound_entry, "Upper Bound")
        else:
            payload["matrix"] = [
                [self.parse_float(cell, f"A[{r + 1},{c + 1}]") for c, cell in enumerate(row)]
                for r, row in enumerate(self.matrix_entries)
            ]
            payload["constants"] = [self.parse_float(cell, f"b[{i + 1}]") for i, cell in enumerate(self.constants_entries)]

        return payload

    def set_entry(self, entry, value):
        if entry is None:
            return
        entry.delete(0, "end")
        entry.insert(0, value)

    def random_linear_system(self, size):
        x_true = np.array([random.randint(-3, 3) or 1 for _ in range(size)], dtype=int)
        matrix = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if i != j:
                    matrix[i, j] = random.randint(-3, 3)
            row_sum = int(np.sum(np.abs(matrix[i, :])))
            matrix[i, i] = row_sum + random.randint(3, 6)
        constants = matrix @ x_true
        return matrix.tolist(), constants.tolist()

    def fill_random(self):
        method = self.controller.selected_method or "newton_raphson"

        if method == "newton_raphson":
            function_text, guess = random.choice([("x**2 - 2", 1.5), ("x**3 - x - 2", 1.5), ("cos(x) - x", 0.7)])
            self.set_entry(self.function_entry, str(function_text))
            self.set_entry(self.initial_guess_entry, str(guess))
            self.set_entry(self.iterations_entry, "20")
            self.set_entry(self.precision_entry, "6")
            return

        if method == "regula_falsi":
            function_text, lower, upper = random.choice([("x**3 - x - 2", 1, 2), ("x**2 - 5", 2, 3), ("sin(x) - 0.5", 0, 1)])
            self.set_entry(self.function_entry, str(function_text))
            self.set_entry(self.lower_bound_entry, str(lower))
            self.set_entry(self.upper_bound_entry, str(upper))
            self.set_entry(self.iterations_entry, "25")
            self.set_entry(self.precision_entry, "6")
            return

        size = int(self.matrix_size.get())
        matrix, constants = self.random_linear_system(size)
        for i in range(size):
            for j in range(size):
                self.set_entry(self.matrix_entries[i][j], str(matrix[i][j]))
            self.set_entry(self.constants_entries[i], str(constants[i]))
        self.set_entry(self.iterations_entry, "20")
        self.set_entry(self.precision_entry, "6")

    def solve_current_method(self):
        method = self.controller.selected_method
        if not method:
            self.error_label.configure(text="Choose a method first.")
            return

        try:
            payload = self.collect_payload(method)
            solver = create_solver(method, payload)
            result = solver.solve()
        except Exception as error:
            self.error_label.configure(text=str(error))
            return

        self.error_label.configure(text="")
        self.controller.last_result = result
        self.controller.last_request = payload
        self.controller.show_frame("ResultFrame")


class ResultFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=PALETTE["bg"])
        self.controller = controller
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        card = ctk.CTkFrame(self, fg_color=PALETTE["surface"], corner_radius=18)
        card.grid(row=0, column=0, padx=34, pady=34, sticky="nsew")
        card.grid_rowconfigure(3, weight=1)
        card.grid_columnconfigure(0, weight=1)

        self.confetti = ConfettiLayer(card, width=940, height=130)
        self.confetti.grid(row=0, column=0, padx=16, pady=(16, 6), sticky="ew")

        self.title_label = ctk.CTkLabel(card, text="Result", font=FONTS["title"], text_color=PALETTE["text_primary"])
        self.title_label.grid(row=1, column=0, pady=(4, 2))

        self.status_label = ctk.CTkLabel(card, text="", font=FONTS["status"], text_color=PALETTE["success"])
        self.status_label.grid(row=2, column=0, pady=(2, 4))

        self.final_label = ctk.CTkLabel(card, text="", font=FONTS["heading"], text_color=PALETTE["text_primary"])
        self.final_label.grid(row=3, column=0, pady=(4, 8), sticky="n")

        self.table = ctk.CTkTextbox(card, font=FONTS["mono"], fg_color="#0F1622", corner_radius=12, text_color="#EAF2FF")
        self.table.grid(row=4, column=0, sticky="nsew", padx=16, pady=(6, 10))
        self.table.configure(state="disabled")

        footer = ctk.CTkFrame(card, fg_color="transparent")
        footer.grid(row=5, column=0, pady=(0, 16))
        ctk.CTkButton(footer, text="Edit Inputs", width=120, fg_color="#2C3C54", hover_color="#3A4D68", command=lambda: controller.show_frame("InputFrame")).pack(side="left", padx=6)
        ctk.CTkButton(footer, text="New Method", width=120, fg_color=PALETTE["accent"], hover_color=PALETTE["accent_hover"], text_color="#04121E", command=lambda: controller.show_frame("SelectionFrame")).pack(side="left", padx=6)

    def on_show(self):
        result = self.controller.last_result
        request = self.controller.last_request or {}
        precision = int(request.get("precision", 6))

        if result is None:
            self.title_label.configure(text="Result")
            self.status_label.configure(text="No result yet", text_color=PALETTE["danger"])
            self.final_label.configure(text="")
            self.set_table([], precision)
            return

        self.title_label.configure(text=result.method_name + " Result")
        if result.converged:
            self.status_label.configure(text="Converged - " + result.message, text_color=PALETTE["success"])
            self.confetti.burst()
        else:
            self.status_label.configure(text="Not Fully Converged - " + result.message, text_color="#F8D66B")
            self.confetti.delete("all")

        self.final_label.configure(text="Final Approximation: " + self.format_value(result.final_value, precision))
        self.set_table(result.rows, precision)

    def set_table(self, rows, precision):
        lines = [f"{'Iter':<8}{'Value':<48}{'Error':<16}", "-" * 80]
        for row in rows:
            lines.append(f"{row.iteration:<8}{self.format_value(row.value, precision):<48}{self.format_error(row.error, precision):<16}")

        self.table.configure(state="normal")
        self.table.delete("1.0", "end")
        self.table.insert("1.0", "\n".join(lines))
        self.table.configure(state="disabled")

    def format_value(self, value, precision):
        if value is None:
            return "-"
        if isinstance(value, list):
            return "[" + ", ".join(f"{float(item):.{precision}f}" for item in value) + "]"
        return f"{float(value):.{precision}f}"

    def format_error(self, value, precision):
        if value is None:
            return "-"
        return f"{float(value):.{precision}f}"

