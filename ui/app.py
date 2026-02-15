import customtkinter as ctk

from .frames import InputFrame, LandingFrame, ResultFrame, SelectionFrame
from .theme import PALETTE


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Numerical Methods Visualizer - Root and Linear Solver")
        self.geometry("1220x760")
        self.minsize(980, 620)
        self.configure(fg_color=PALETTE["bg"])

        container = ctk.CTkFrame(self, fg_color=PALETTE["bg"])
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.selected_method = None
        self.last_result = None
        self.last_request = None

        self.frames = {}
        for frame_class in (LandingFrame, SelectionFrame, InputFrame, ResultFrame):
            frame = frame_class(container, self)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[frame_class.__name__] = frame

        self.show_frame("LandingFrame")

    def show_frame(self, frame_name):
        frame = self.frames[frame_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()
