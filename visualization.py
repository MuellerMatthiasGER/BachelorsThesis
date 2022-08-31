import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb

delta = 0.25
hue = np.arange(0, 360, delta)
lightness = np.arange(0, 100, delta)
Hue, Lightness = np.meshgrid(hue, lightness)

levels_show = np.linspace(np.e ** -0.5, 0.95, num=3)
linewidths_show = [1.5, 0.3, 0.3]
levels_compare = levels_show[0:1]
linewidths_compare = linewidths_show[0:1]

background = np.empty(len(hue) * len(lightness) * 3)
for y, l in enumerate(lightness):
    for x, h in enumerate(hue):
        r, g, b = hls_to_rgb(h / 360.0, l / 100.0, 1)
        idx = (y * len(hue) + x) * 3
        background[idx] = r
        background[idx + 1] = g
        background[idx + 2] = b

background = background.reshape(len(lightness), len(hue), 3)

def translate_color(color_label: str):
    if color_label == 'rot':
        return 'red'
    elif color_label == 'braun':
        return 'brown'
    elif color_label == 'gelb':
        return 'yellow'
    elif color_label == 'grün':
        return 'green'
    elif color_label == 'blattgrün':
        return 'darkgreen'
    elif color_label == 'türkis':
        return 'turquoise'
    elif color_label == 'blau':
        return 'blue'
    elif color_label == 'marinblau':
        return 'navy'
    elif color_label == 'lila':
        return 'purple'
    elif color_label == 'pink':
        return 'deeppink'
    elif color_label == 'bordeaux':
        return 'darkred'
    elif color_label in ['orange', 'olive', 'indigo']:
        return color_label
    else:
        return 'black'

def _plot(ax: plt.Axes, row: pd.Series, contour_color: str, linestyles: str, linewidths=linewidths_show, levels=levels_show, hue_offset=0):
    Values = np.exp(-0.5 * (((Hue - (row['hc'] + hue_offset)) / row['hr']) ** 2 + 0 + ((Lightness - row['lc']) / row['lr']) ** 2))
    ax.contour(Hue, Lightness, Values, colors=contour_color, levels=levels, linewidths=linewidths, linestyles=linestyles)

    if hue_offset == 0:
        if row['hc'] - row['hr'] < 0:
            _plot(ax, row, contour_color, linestyles, linewidths, levels, 360)
        if row['hc'] + row['hr'] > 360:
            _plot(ax, row, contour_color, linestyles, linewidths, levels, -360)

def show(model: pd.DataFrame, base_model: pd.DataFrame = None, xlim=[0, 360]):
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)

    linewidths = linewidths_show
    levels = levels_show

    if base_model is not None:
        linewidths = linewidths_compare
        levels = levels_compare

        for colorname, row in base_model.iterrows():
            contour_color = translate_color(colorname)

            _plot(ax, row, contour_color, 'dotted', linewidths=linewidths, levels=levels)

    for colorname, row in model.iterrows():
        contour_color = translate_color(colorname)

        _plot(ax, row, contour_color, 'solid', linewidths=linewidths, levels=levels)


    plt.imshow(background, extent=[0, 360, 0, 100], origin='lower')
    plt.xticks(np.arange(0, 361, 30))
    plt.xlim(xlim)
    
    ax.set_xlabel("Hue (Degrees)")
    ax.set_ylabel("Lightness (%)")

    plt.savefig("output.png",bbox_inches='tight')
    plt.show()
