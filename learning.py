import pandas as pd
import numpy as np
import itertools
from scipy.stats import circmean
from scipy.optimize import minimize
import logging

from colors import Color
import model 

aimed_difference = 0.02
convergence_tolerance = 0.01
add_all_distracting_categories = False # standard, see Paper

accumulated_number = 15

log_columns = ['hc', 'hr', 'sc', 'sr', 'lc', 'lr', 'acc0', 'acc1', 'score0', 'score1', 'change_borders']

# ======================================
# ============== Goals =================
# ======================================

def generate_naive_goal(model: pd.DataFrame, color_index: int, target_category: str):
    return _generate_goal(model, color_index, target_category, metric = 'acc')

def generate_context_sensitive_goal(model: pd.DataFrame, color_index: int, target_category: str):
    return _generate_goal(model, color_index, target_category, metric = 'score')

def _generate_goal(model: pd.DataFrame, color_index: int, target_category: str, metric: str = 'score'):
    goal = {'color_index': color_index, 'target_category': target_category}
    
    if add_all_distracting_categories:
        goal['distracting_categories'] = list(model.index[model[f'{metric}{color_index}'] > model.at[target_category, f'{metric}{color_index}']])
    else:
        cur_category = model[f'{metric}{color_index}'].idxmax()
        if target_category != cur_category:
            goal['distracting_categories'] = [cur_category]
        else:
            goal['distracting_categories'] = []
    
    return goal

# ======================================
# =========== Statistical ==============
# ======================================

def execute_statistical_learning(model: pd.DataFrame, color_name, h, s, l, change_limit=float('inf')):
    color = model.loc[color_name]

    hues = np.full(accumulated_number, color['hc'])
    hues[0] = h
    new_hc = circmean(hues, high=360, low=0)
    delta = (new_hc - color['hc']) % 360
    if delta > 180:
        delta -= 360
    color['hc'] = new_hc
    color['hr'] = np.sqrt(((accumulated_number - 1) * (color['hr'] ** 2 + delta ** 2) + min((h - new_hc) ** 2, (h - 360 - new_hc) ** 2, (h + 360 - new_hc) ** 2)) / accumulated_number)
    new_sc = ((accumulated_number - 1) * color['sc'] + s) / accumulated_number
    delta = new_sc - color['sc']
    color['sc'] = new_sc
    color['sr'] = np.sqrt(((accumulated_number - 1) * (color['sr'] ** 2 + delta ** 2) + (s - new_sc) ** 2) / accumulated_number)
    new_lc = ((accumulated_number - 1) * color['lc'] + l) / accumulated_number
    delta = new_lc - color['lc']
    color['lc'] = new_lc
    color['lr'] = np.sqrt(((accumulated_number - 1) * (color['lr'] ** 2 + delta ** 2) + (l - new_lc) ** 2) / accumulated_number)

    change_border = calc_boder_change(model.loc[color_name], color)

    if change_border < change_limit:
        logging.info(f"Border Change: {change_border}")
        model.loc[color_name] = color
    else:
        logging.info(f"The change limit of {change_limit} was exceeded. Model not changed.")
        print(f"The change limit of {change_limit} was exceeded. Model not changed.")

def calc_boder_change(prot_old, prot_new):
    h0_old = (prot_old['hc'] - prot_old['hr']) % 360
    h1_old = (prot_old['hc'] + prot_old['hr']) % 360
    h0_new = (prot_new['hc'] - prot_new['hr']) % 360
    h1_new = (prot_new['hc'] + prot_new['hr']) % 360

    border_change = 0
    border_change += min(np.abs(h0_new - h0_old), np.abs(h0_new - 360 - h0_old), np.abs(h0_new + 360 - h0_old)) ** 2
    border_change += min(np.abs(h1_new - h1_old), np.abs(h1_new - 360 - h1_old), np.abs(h1_new + 360 - h1_old)) ** 2
    border_change += (max(min(prot_new['sc'] - prot_new['sr'], 100), 0) - max(min(prot_old['sc'] - prot_old['sr'], 100), 0))  ** 2
    border_change += (max(min(prot_new['sc'] + prot_new['sr'], 100), 0) - max(min(prot_old['sc'] + prot_old['sr'], 100), 0))  ** 2
    border_change += (max(min(prot_new['lc'] - prot_new['lr'], 100), 0) - max(min(prot_old['lc'] - prot_old['lr'], 100), 0))  ** 2
    border_change += (max(min(prot_new['lc'] + prot_new['lr'], 100), 0) - max(min(prot_old['lc'] + prot_old['lr'], 100), 0))  ** 2

    return border_change

# ======================================
# =========== Minimization =============
# ======================================

def adjust(x):
    global adjustedX
    if np.array_equal(x, adjustedX):
        return
    adjustedX = x
    
    adjust_model['hc'] = adjust_model.apply(lambda prot: (prot['hc_real'] + x[int(prot['idx'] * 6)]) % 360, axis=1)
    adjust_model['hr'] = adjust_model.apply(lambda prot: max(min(prot['hr_real'] + x[int(prot['idx'] * 6 + 1)], 180), 1), axis=1)
    adjust_model['h0'] = adjust_model.apply(lambda prot: (prot['hc'] - prot['hr']) % 360, axis=1)
    adjust_model['h1'] = adjust_model.apply(lambda prot: (prot['hc'] + prot['hr']) % 360, axis=1)
    adjust_model['sc'] = adjust_model.apply(lambda prot: max(min(prot['sc_real'] + x[int(prot['idx'] * 6 + 2)], 100), 0), axis=1)
    adjust_model['sr'] = adjust_model.apply(lambda prot: max(min(prot['sr_real'] + x[int(prot['idx'] * 6 + 3)], 100), 1), axis=1)
    adjust_model['s0'] = adjust_model.apply(lambda prot: max(min(prot['sc'] - prot['sr'], 100), 0), axis=1)
    adjust_model['s1'] = adjust_model.apply(lambda prot: max(min(prot['sc'] + prot['sr'], 100), 0), axis=1)
    adjust_model['lc'] = adjust_model.apply(lambda prot: max(min(prot['lc_real'] + x[int(prot['idx'] * 6 + 4)], 100), 0), axis=1)
    adjust_model['lr'] = adjust_model.apply(lambda prot: max(min(prot['lr_real'] + x[int(prot['idx'] * 6 + 5)], 100), 1), axis=1)
    adjust_model['l0'] = adjust_model.apply(lambda prot: max(min(prot['lc'] - prot['lr'], 100), 0), axis=1)
    adjust_model['l1'] = adjust_model.apply(lambda prot: max(min(prot['lc'] + prot['lr'], 100), 0), axis=1)

    model.fit_colors(colors, used_model=adjust_model)

def constraint_metric(x, color_index, target_category, distracting_category, metric: str = 'score'):
    adjust(x)
    return adjust_model.at[target_category, f'{metric}{color_index}'] - adjust_model.at[distracting_category, f'{metric}{color_index}'] - aimed_difference

def calc_border_change(prot):
    sum = 0
    # add hue border change
    for i in [0,1]:
        sum += min(np.abs(prot[f'h{i}'] - prot[f'h{i}_real']), np.abs(prot[f'h{i}'] - 360 - prot[f'h{i}_real']), np.abs(prot[f'h{i}'] + 360 - prot[f'h{i}_real'])) ** 2
    # add saturation and lightness border
    for (dimension, i) in itertools.product(['s', 'l'], [0, 1]):
        sum += (prot[f'{dimension}{i}'] - prot[f'{dimension}{i}_real']) ** 2
    
    return sum

def change(x):
    adjust(x)

    adjust_model['change_borders'] = adjust_model.apply(calc_border_change, axis=1)
    return adjust_model['change_borders'].sum()

def execute_naive_learning(model: pd.DataFrame, colors_input: list[Color], goals: dict, change_limit=float('inf')):
    return _execute_learning(model, colors_input, goals, metric='acc', change_limit=change_limit)

def execute_context_sensitive_learning(model: pd.DataFrame, colors_input: list[Color], goals: dict, change_limit=float('inf')):
    return _execute_learning(model, colors_input, goals, metric='score', change_limit=change_limit)

def _execute_learning(model: pd.DataFrame, colors_input: list[Color], goals: dict, metric: str, change_limit: float):
    global colors 
    colors = colors_input
    
    involved_prots = set()
    for goal in goals:
        involved_prots.add(goal['target_category'])
        involved_prots.update(goal['distracting_categories'])
    mask =  model.index.to_series().isin(involved_prots)

    global adjust_model
    adjust_model = model.loc[mask, ['hc', 'hr', 'sc', 'sr', 'lc', 'lr']].copy()
    adjust_model['idx'] = range(0, len(adjust_model))
    adjust_model['hc_real'] = adjust_model['hc']
    adjust_model['hr_real'] = adjust_model['hr']
    adjust_model['sc_real'] = adjust_model['sc']
    adjust_model['sr_real'] = adjust_model['sr']
    adjust_model['lc_real'] = adjust_model['lc']
    adjust_model['lr_real'] = adjust_model['lr']
    adjust_model['h0_real'] = adjust_model.apply(lambda prot: (prot['hc_real'] - prot['hr_real']) % 360, axis=1)
    adjust_model['h1_real'] = adjust_model.apply(lambda prot: (prot['hc_real'] + prot['hr_real']) % 360, axis=1)
    adjust_model['s0_real'] = adjust_model.apply(lambda prot: max(min(prot['sc_real'] - prot['sr_real'], 100), 0), axis=1)
    adjust_model['s1_real'] = adjust_model.apply(lambda prot: max(min(prot['sc_real'] + prot['sr_real'], 100), 0), axis=1)
    adjust_model['l0_real'] = adjust_model.apply(lambda prot: max(min(prot['lc_real'] - prot['lr_real'], 100), 0), axis=1)
    adjust_model['l1_real'] = adjust_model.apply(lambda prot: max(min(prot['lc_real'] + prot['lr_real'], 100), 0), axis=1)
    adjust_model['change_borders'] = 0

    x0 = np.zeros(len(involved_prots) * 6)
    constraints = []
    for goal in goals:
        for distracting_category in goal['distracting_categories']:
            function = lambda x: constraint_metric(x, goal['color_index'], goal['target_category'], distracting_category, metric)
            constraints.append({'type': 'ineq', 'fun': function})

    global adjustedX
    adjustedX = np.ones(len(involved_prots) * 6) # just to differ from x0 for executing adjust() once 
    adjust(x0)

    print(adjust_model[log_columns])
    logging.info(adjust_model[log_columns])

    result = minimize(change, x0, constraints=constraints, tol=convergence_tolerance)
    if result.success:
        if adjust_model['change_borders'].sum() < change_limit:
            print(adjust_model[log_columns])
            logging.info(adjust_model[log_columns])
            model.loc[mask, ['hc', 'hr', 'sc', 'sr', 'lc', 'lr']] = adjust_model[['hc', 'hr', 'sc', 'sr', 'lc', 'lr']]
        else:
            logging.info(f"The change limit of {change_limit} was exceeded. Model not changed.")
            print(f"The change limit of {change_limit} was exceeded. Model not changed.")    
    else:
        logging.info("Optimization not successful. Model not changed.")
        print("Optimization not successful. Model not changed.")
        print(result)