import random
import tkinter as tk


class ConfettiLayer(tk.Canvas):
    COLORS = ["#34D399", "#F472B6", "#FBBF24", "#60A5FA", "#F87171", "#A78BFA"]

    def __init__(self, parent, width=840, height=160):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg="#111821",
            bd=0,
            highlightthickness=0,
        )
        self._pieces = []
        self._running = False
        self._width = width
        self._height = height

    def burst(self):
        self.delete("all")
        self._pieces = []
        for _ in range(85):
            x = random.uniform(0, self._width)
            y = random.uniform(-120, -10)
            size = random.uniform(4, 8)
            speed = random.uniform(1.4, 3.6)
            drift = random.uniform(-1.2, 1.2)
            color = random.choice(self.COLORS)
            item = self.create_oval(x, y, x + size, y + size, fill=color, outline="")
            self._pieces.append({"id": item, "x": x, "y": y, "size": size, "speed": speed, "drift": drift})

        self._running = True
        self.after(16, self._animate)

    def _animate(self):
        if not self._running:
            return

        alive = 0
        for piece in self._pieces:
            piece["x"] += piece["drift"]
            piece["y"] += piece["speed"]
            x = piece["x"]
            y = piece["y"]
            size = piece["size"]
            self.coords(piece["id"], x, y, x + size, y + size)
            if y < self._height + 20:
                alive += 1

        if alive == 0:
            self._running = False
            return

        self.after(16, self._animate)
