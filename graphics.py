"""This script is used to create artificial program runs to create graphics for the thesis."""

import numpy as np
import matplotlib as plt
import itertools

import model
import learning
from logdecoder import decode_log, determine_errors
from colors import Color

def robustness_statistical():
    for _ in range(20):
        model.execute_statistical_learning(model.model, 'grün', 160, 100, 50)

    model.show_comparision()

def robustness_naive():
    colors = [Color(0, 100, 50), Color(35, 100, 50)]
    model.fit_colors(colors)
    goal = learning.generate_naive_goal(model.model, 1, 'gelb')

    model.execute_naive_learning(model.model, colors, [goal])
    model.show_comparision()

def robustness_sensitive_pos():
    colors = [Color(0, 100, 50), Color(35, 100, 50)]
    model.fit_colors(colors)
    goal = learning.generate_context_sensitive_goal(model.model, 1, 'gelb')

    model.execute_context_sensitive_learning(model.model, colors, [goal])
    model.show_comparision()

def robustness_sensitive_neg():
    colors = [Color(0, 100, 50), Color(35, 100, 50)]
    model.fit_colors(colors)
    goal = learning.generate_context_sensitive_goal(model.model, 1, 'gelb')

    model.execute_context_sensitive_learning(model.model, colors, [goal])
    model.show_comparision()

def make_subplot(title, collection, start_index, end_index, plot_rows, plot_cols, plot_number, ylim):
    collection_entries = [decode_log(id) for id in collection]
    collection_errors = [determine_errors(id) for id in collection]

    x = range(0, end_index)
    y = [0] * end_index

    for entry_list, error_list in zip(collection_entries, collection_errors):
        for index, entry in enumerate(entry_list[:end_index]):
            if error_list[index]:
                y[index] += 750
            else:
                y[index] += entry.change_borders

    y = [value / len(collection_entries) for value in y]

    print(sum(y) / len(y))

    m, t = np.polyfit(x[start_index : end_index], y[start_index : end_index], deg=1)
    print(m, t)

    plt.subplot(plot_rows, plot_cols, plot_number)
    plt.plot(x, y, label="Average: %.1f" % (sum(y) / len(y)))
    plt.axline(xy1=(0, t), slope=m, color='r', label="Slope: %.2f" % m)
    plt.legend()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Average Border Change")
    plt.ylim((0, ylim))

def make_error_subplot(collection, start_index, end_index, plot_rows, plot_cols, plot_number, ylim):
    collection_errors = [determine_errors(id) for id in collection]

    x = range(0, end_index)
    y = [0] * end_index

    for error_list in collection_errors:
        for index, error in enumerate(error_list[:end_index]):
            if error:
                y[index] += 1

    y = [value / len(collection_errors) for value in y]

    m, t = np.polyfit(x[start_index : end_index], y[start_index : end_index], deg=1)

    print(m, t)

    plt.subplot(plot_rows, plot_cols, plot_number)
    plt.plot(x, y, color='orange')
    plt.axline(xy1=(0, t), slope=m, color='r', label="Slope: %.2e" % m)
    plt.legend()
    plt.ylim((0, ylim))
    plt.xlabel("Iterations")
    plt.ylabel("Percentage of Failed Adaptions")

def make_prediction_subplot(title, collection, start_index, end_index, plot_rows, plot_cols, plot_number, ylim):
    collection_entries = [decode_log(id) for id in collection]

    x = range(0, end_index)
    y = [0] * end_index

    for entry_list in collection_entries:
        for index, entry in enumerate(entry_list[:end_index]):
            if entry.correct_pred:
                y[index] += 1

    y = [value / len(collection_entries) for value in y]

    print(sum(y) / len(y))

    m, t = np.polyfit(x[start_index : end_index], y[start_index : end_index], deg=1)
    print(m, t)

    plt.subplot(plot_rows, plot_cols, plot_number)
    plt.plot(x, y, color='green', label="Average: %.3f" % (sum(y) / len(y)))
    plt.axline(xy1=(0, t), slope=m, color='r', label="Slope: %.2e" % m)
    plt.legend()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Correct Prediction by PRAGR")
    plt.ylim((0, ylim))

def eval_border_change():
    plt.rcParams["figure.figsize"] = (18, 8)

    make_subplot("Statistical Approach", ['10%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 2, 3, 1, 300)
    make_subplot("Naive Approach", ['20%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 2, 3, 2, 300)
    make_subplot("Context-Sensitive Approach", ['30%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 2, 3, 3, 300)
    make_error_subplot(['10%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 2, 3, 4, 0.3)
    make_error_subplot(['20%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 2, 3, 5, 0.3)
    make_error_subplot(['30%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 2, 3, 6, 0.3)

    plt.show()

def eval_prediction():
    plt.rcParams["figure.figsize"] = (12, 4)

    make_prediction_subplot("Statistical Approach", ['10%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 1, 2, 1, 1)
    make_prediction_subplot("Context-Sensitive Approach", ['30%02d%s' % pair for pair in itertools.product(range(1,13), ['a', 'b', 'c'])], 0, 60, 1, 2, 2, 1)

    plt.show()

def eval_border_change_no_lower():
    plt.rcParams["figure.figsize"] = (18, 8)

    make_subplot("Statistical Approach", ['10%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 2, 3, 1, 210)
    make_subplot("Naive Approach", ['20%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 2, 3, 2, 210)
    make_subplot("Context-Sensitive Approach", ['30%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 2, 3, 3, 210)
    make_error_subplot(['10%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 2, 3, 4, 0.175)
    make_error_subplot(['20%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 2, 3, 5, 0.175)
    make_error_subplot(['30%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 2, 3, 6, 0.175)

    plt.show()

def eval_prediction_no_lower():
    plt.rcParams["figure.figsize"] = (12, 4)

    make_prediction_subplot("Statistical Approach", ['10%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 1, 2, 1, 1)
    make_prediction_subplot("Context-Sensitive Approach", ['30%02d%s' % pair for pair in itertools.product(range(1,13), ['x', 'y', 'z'])], 0, 60, 1, 2, 2, 1)

    plt.show()


if __name__ == '__main__':
    pass