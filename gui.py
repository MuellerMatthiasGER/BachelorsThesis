import random
import tkinter as tk
from tkinter import ttk
import logging
import random

from colors import *
import model
from logdecoder import Entry


class App(tk.Tk):
    def __init__(self, log_id, open_log_mode):
        super().__init__()
        self.title("Colorful")

        self.canvas_size = 400
        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.description = tk.StringVar()
        label = ttk.Label(self, textvariable=self.description, wraplength=self.canvas_size)
        label.pack(padx=10, fill='x', expand=True)

        self.input = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.input)
        self.entry.pack(padx=10, pady=10, fill='x', expand=True)

        self.log_id = log_id
        self.learn_mode = 'context-sensitive'
        logging.basicConfig(filename=f'colorful/logs/log{log_id}.log', filemode=open_log_mode, level=logging.INFO, format='%(message)s', encoding='utf-8')

        self.validation_entries: list[Entry] = []
        self.validation_iterator = iter(self.validation_entries)
        self.correct_validated = 0

    def new_validation(self, validation_entries: list[Entry] = None):
        if validation_entries is not None:
            logging.info('==============')
            logging.info('= Validation =')
            logging.info('==============')
            random.shuffle(validation_entries)
            validation_number = min(len(validation_entries), 25)
            self.validation_entries = validation_entries[:validation_number]
            self.validation_iterator = iter(self.validation_entries)

        entry = next(self.validation_iterator, None)

        if entry is None:
            validation_rate = self.correct_validated / len(self.validation_entries)
            logging.info(f"Validation Rate: {validation_rate}")
            self.destroy()
            return
        
        self.colors = entry.colors
        target_color = self.colors[entry.target_index]
        random.shuffle(self.colors)
        self.target_index = self.colors.index(target_color)

        self.draw(draw_indication_arrow=False)

        self.log_colors()
        logging.info(f"Input: {entry.input}")

        self.description.set(f'Welche Scheibe ist {entry.input.upper()}?\nGebe "links" (1), "rechts" (2) oder "keine" (0) ein.')
        self.entry.bind('<Return>', self.new_validation_entry)

    def new_validation_entry(self, _):
        text = self.input.get()
        self.input.set('')

        if text == 'links':
            indicated = 0
        elif text == 'rechts':
            indicated = 1
        elif text == 'keine':
            indicated = -1
        elif text in ['0', '1', '2']:
            indicated = int(text) - 1
        else:
            indicated = None

        if indicated is not None:
            correct = self.target_index == indicated
            if correct:
                self.correct_validated += 1
            logging.info(f"Validation: {correct}")
            logging.info("---")
            self.new_validation()

    def new_learn(self):
        self.colors = two_adjacent_colors(model.model, radius_portion=1.2)
        self.target_index = random.randint(0, len(self.colors) - 1)
        self.draw()
        model.fit_colors(self.colors)

        self.log_colors()
        print(model.model)

        best_desc = model.best_descriptions_naive(len(self.colors))
        for i in range(len(self.colors)):
            print(best_desc[i])
        print("---")

        best_desc = model.best_descriptions(len(self.colors))
        for i in range(len(self.colors)):
            print(best_desc[i])
        print("---")

        self.description.set(f"Mit welcher Farbe würdest du die {'LINKE' if self.target_index == 0 else 'RECHTE'} Scheibe beschreiben, um sie eindeutig zu identifizieren und von der anderen Scheibe abzugrenzen?")
        self.entry.bind('<Return>', self.new_learn_entry)

    def new_learn_entry(self, _):
        text = self.input.get().lower()
        self.input.set('')
        if text == 'save!':
            model.save_model(self.log_id)
            logging.info("<< Model Saved! >>")
        elif text == 'show!':
            model.show_model()
        elif text == 'compare!':
            model.show_comparision()
        elif self.set_learn_mode(text):
            pass
        elif text == '?':
            model.indistinguishable_colors(self.target_index)
            logging.info(f"Indistinguishable!")
            logging.info("---")
            self.new_learn()
        else:
            logging.info(f"Input: {text}")

            if self.learn_mode == 'statistical':
                model.statistical_learning(text, self.colors, self.target_index)
            elif self.learn_mode == 'naive':
                color_labels = [None] * 2
                color_labels[self.target_index] = text
                model.naive_learning(self.colors, color_labels=color_labels)
            elif self.learn_mode == 'context-sensitive':
                color_labels = [None] * 2
                color_labels[self.target_index] = text
                model.context_sensitive_learning(self.colors, color_labels=color_labels)
            else:
                print('Unknown learn mode! No learning executed!')
            
            if model.colorname_exists(text):
                logging.info("---")
                self.new_learn()

    def set_learn_mode(self, learn_mode):
        if learn_mode in ['statistical!', 'naive!', 'context-sensitive!']:
            self.learn_mode = learn_mode.replace('!', '')
            print(f"Learn mode set to: {self.learn_mode}")
            logging.info(f"<< Learn mode set to: {self.learn_mode} >>")
            return True

        return False

    def log_colors(self):
        for i, c in enumerate(self.colors):
            logging.info(f"Color{i}: {c}")
        logging.info(f"Target Index: {self.target_index}")

    def draw(self, draw_indication_arrow=True):
        self.canvas.delete('all')
        self.canvas.create_oval(50, 50, 150, 150, width=2, fill=self.colors[0].html())
        self.canvas.create_oval(250, 50, 350, 150, width=2, fill=self.colors[1].html())
        if len(self.colors) > 2:
            self.canvas.create_oval(50, 250, 150, 350, width=2, fill=self.colors[2].html())
        if len(self.colors) > 3:
            self.canvas.create_oval(250, 250, 350, 350, width=2, fill=self.colors[3].html())

        # draw indication arrow
        if draw_indication_arrow and len(self.colors) == 2:
            arrowX = 100 if self.target_index == 0 else 300
            self.canvas.create_line(arrowX, 250, arrowX, 180, width=3, arrow='last', arrowshape=(16, 20, 6))