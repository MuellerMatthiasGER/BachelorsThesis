import logging
import tkinter as tk
from tkinter import ttk
import random

import model
import logdecoder
import visualization


"""
This script can be used to simulate the inputs of a user to other learning algorithms or to other parameters.

origin_id: is the id of the log that is used to extract the color from
learn_mode: determines which learning algorithm is used for the simulation. Possible values are 
        'statistical'       (the resulting log is origin_id + 1000)
        'naive'             (the resulting log is origin_id + 2000)
        'context-sensitive' (the resulting log is origin_id + 3000)
name_appendix: is a string added to the log file name
double: if activated, the same inputs are presented twice to the algorithm to check
shuffle: if activated, the scenarios (target + distractor color) stored in the log are shuffled
        and are applied in random order to the learning algorithm
show_steps: if activated, the change in the model and the two input colors are shown after each input
show_model: if actived, the model is shown at the end compared to the base model from the beginning
change_limit: is the highest allowed change per step. 
        If it is exceeded, the description is considered as an outlier and the model is not adjusted.
        Can be disabled by entering: float('inf')
"""


# ---------------------- #
origin_id = 3
learn_mode = 'context-sensitive'
name_appendix = '_appendix'
double = False
shuffle = False
show_steps = True
show_model = False
change_limit = 750
# ---------------------- #



class DemoGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Colorful")

        self.canvas_size = 400
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.description = tk.StringVar()
        label = ttk.Label(self, textvariable=self.description, wraplength=self.canvas_size)
        label.pack(padx=10, fill='x', expand=True)

    def draw(self, colors, target_index, input):
        self.description.set(input)

        self.canvas.delete('all')
        self.canvas.create_oval(50, 50, 150, 150, width=2, fill=colors[0].html())
        self.canvas.create_oval(250, 50, 350, 150, width=2, fill=colors[1].html())

        # draw indication arrow
        arrowX = 100 if target_index == 0 else 300
        self.canvas.create_line(arrowX, 250, arrowX, 180, width=3, arrow='last', arrowshape=(16, 20, 6))


def simulate():
    if learn_mode == 'statistical':
        log_id = origin_id + 1000
    elif learn_mode == 'naive':
        log_id = origin_id + 2000
    elif learn_mode == 'context-sensitive':
        log_id = origin_id + 3000
    else:
        raise Exception("Error: Invalid Learning Mode!")

    fileh = logging.FileHandler(f'colorful/logs/log{log_id}{name_appendix}.log', 'w', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.INFO)

    logging.info(f"<< Learn Mode: {learn_mode} >>")
    logging.info(f"<< Change Limit set to {change_limit} >>")

    if show_steps:
        gui = DemoGui()

    entries = logdecoder.decode_log(origin_id)[:60]
    if shuffle:
        random.shuffle(entries)
    if double:
        entries += entries

    for index, entry in enumerate(entries):
        print(f"Considering entry {index}")

        colors = entry.colors
        target_index = entry.target_index
        model.fit_colors(colors, used_model=model.model)

        for i, c in enumerate(colors):
            logging.info(f"Color{i}: {c}")
        logging.info(f"Target Index: {target_index}")
        print(f"Target Index: {target_index}")

        input = entry.input
        logging.info(f"Input: {input}")
        print(f"Input: {input}")

        if show_steps:
            model_before = model.model[['hc', 'hr', 'sc', 'sr', 'lc', 'lr']].copy()
        
        if learn_mode == 'statistical':
            model.statistical_learning(input, colors, target_index, change_limit=change_limit)
        elif learn_mode == 'naive':
            color_labels = [None] * 2
            color_labels[target_index] = input
            model.naive_learning(colors, color_labels=color_labels, change_limit=change_limit)
        elif learn_mode == 'context-sensitive':
            color_labels = [None] * 2
            color_labels[target_index] = input
            model.context_sensitive_learning(colors, color_labels=color_labels, change_limit=change_limit)
        else:
            print('Unknown learn mode! No learning executed!')
        
        logging.info("---")

        if show_steps:
            gui.draw(colors, target_index, input)
            visualization.show(model.model, model_before)

    if show_model:
        visualization.show(model.model, model.base_model)


if __name__ == '__main__':
    simulate()
