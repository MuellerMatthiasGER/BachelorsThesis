import colorsys
import random
import math
import pandas as pd

class Color:
    def __init__(self, h=0, s=100, l=50):
        self.h = h
        self.s = s
        self.l = l

    def html(self):
        (r, g, b) = colorsys.hls_to_rgb(self.h / 360, self.l / 100, self.s / 100)
        return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

    def hsl(self):
        return (self.h, self.s, self.l)

    def __str__(self):
        return f"Color(h: {self.h}, s: {self.s}, l: {self.l})"


def two_adjacent_colors_naive(step=20):
    h1 = random.randint(0, 359)
    h2 = (h1 + step) % 360
    colors = [Color(h1, 100, 50), Color(h2, 100, 50)]
    return colors


def two_adjacent_colors(model: pd.DataFrame, radius_portion=1.2):
    color_index = model.index[random.randint(0, model.shape[0] - 1)]
    hc = model.at[color_index, 'hc']
    hr = model.at[color_index, 'hr']
    h_deviation = random.uniform(-1, 1)
    h1 = int(hc + hr * h_deviation) % 360

    go_left = bool(random.getrandbits(1))
    if go_left:
        h_deviation -= radius_portion
        h2 = int(hc + hr * h_deviation - model.at[color_index, 'add_diversity']) % 360
    else:
        h_deviation += radius_portion
        h2 = int(hc + hr * h_deviation + model.at[color_index, 'add_diversity']) % 360

    color1 = Color(h1, 100, random.randint(30, 60))
    color2 = Color(h2, 100, random.randint(30, 60))
    colors = [color1, color2]
    random.shuffle(colors)

    return colors
