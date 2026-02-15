from io import BytesIO

import customtkinter as ctk
from matplotlib import pyplot as plt
from PIL import Image


def formula_image(latex_formula, fg_color="#9BD8FF"):
    figure = plt.figure(figsize=(5.1, 0.8), dpi=140)
    figure.patch.set_alpha(0)
    axis = figure.add_subplot(111)
    axis.axis("off")
    axis.text(0.0, 0.5, f"${latex_formula}$", fontsize=16, color=fg_color, va="center")

    buffer = BytesIO()
    figure.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    buffer.seek(0)

    image = Image.open(buffer).convert("RGBA")
    return ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
