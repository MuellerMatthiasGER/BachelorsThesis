import numpy as np
import pandas as pd
import logging

from colors import *
from visualization import show
from learning import execute_statistical_learning, execute_context_sensitive_learning, execute_naive_learning, generate_naive_goal, generate_context_sensitive_goal


model = pd.read_csv('colorful/knowledge_base/kb_colors_preset.csv', index_col='colorname')
model['add_diversity'] = 0
base_model = model.copy()

alpha = 0.3
lower_bound = 0.02

def acc(prototype: pd.Series, color: Color):
    h, s, l = color.hsl()

    h_part = 0
    if prototype.notna()['hc']:
        h_part = min((h - prototype['hc']) ** 2, (h + 360 - prototype['hc']) ** 2, (h - 360 - prototype['hc']) ** 2) / (prototype['hr'] ** 2)
    s_part = ((s - prototype['sc']) / prototype['sr']) ** 2
    l_part = ((l - prototype['lc']) / prototype['lr']) ** 2

    return max(np.exp(-0.5 * (h_part + s_part + l_part)), lower_bound)

def dp(prototype: pd.Series, index, n_objects):
    sum = 0
    for i in range(n_objects):
        sum += prototype[f'acc{i}']
    return prototype[f'acc{index}'] / sum

def fit_colors(colors: list[Color], used_model=model):
    for i, color in enumerate(colors):
        # add acceptability
        used_model[f'acc{i}'] = used_model.apply(lambda x: acc(x, color), axis=1)
    for i in range(len(colors)):
        # add discriminatory power
        used_model[f'dp{i}'] = used_model.apply(lambda x: dp(x, i, len(colors)), axis=1)
        # calc score
        used_model[f'score{i}'] = used_model.apply(lambda x: alpha * x[f'acc{i}'] + (1 - alpha) * x[f'dp{i}'], axis=1)

def best_descriptions_naive(color_number):
    return [model[f'acc{i}'].idxmax() for i in range(color_number)]

def best_descriptions(color_number):
    return [model[f'score{i}'].idxmax() for i in range(color_number)]

def colorname_exists(color_name):
    if color_name not in model.index:
        print(f"Colorname {color_name} is unknown!")
        return False
    return True

def indistinguishable_colors(target_index):
    colorname = model[f'acc{target_index}'].idxmax()
    model.at[colorname, 'add_diversity'] += 5 

def statistical_learning(color_name, colors: list[Color], target_index, change_limit=float('inf')):
    if not colorname_exists(color_name):
        return

    naive_desc = best_descriptions_naive(len(colors))[target_index]
    context_desc = best_descriptions(len(colors))[target_index]
    logging.info(f"Naive Prediction Correct: {naive_desc == color_name}")
    logging.info(f"Context Prediction Correct: {context_desc == color_name}")

    execute_statistical_learning(model, color_name, colors[target_index].h, colors[target_index].s, colors[target_index].l, change_limit)

def naive_learning(colors: list[Color], color_labels: list[str], change_limit=float('inf')):
    goals = []
    for index, label in enumerate(color_labels):
        if label is not None:
            if not colorname_exists(label):
                return

            goal = generate_naive_goal(model, index, label)
            if goal['distracting_categories']:
                goals.append(goal)
    
    if goals:
        execute_naive_learning(model, colors, goals, change_limit)

def context_sensitive_learning(colors: list[Color], color_labels: list[str], change_limit=float('inf')):
    goals = []
    for index, label in enumerate(color_labels):
        if label is not None:
            if not colorname_exists(label):
                return

            goal = generate_context_sensitive_goal(model, index, label)
            if goal['distracting_categories']:
                goals.append(goal)
    
    if goals:
        execute_context_sensitive_learning(model, colors, goals, change_limit)

def save_model(id):
    model[['hc', 'hr', 'sc', 'sr', 'lc', 'lr']].to_csv(f'colorful/knowledge_base/kb_colors_learned_{id}.csv')
    print("Model saved!")

def show_model():
    show(model)

def show_comparision():
    show(model, base_model)

def reload_model():
    global model
    model = pd.read_csv('colorful/knowledge_base/kb_colors_preset.csv', index_col='colorname')