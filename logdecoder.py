import re
from colors import Color

class Entry:
    def __init__(self):
        self.colors = []
        self.target_index = -1
        self.input = None
        self.validation = None
        self.change_borders = 0
        self.correct_pred = False

def decode_log(log_id):
    entries = []
    with open(f'colorful/logs/log{log_id}.log', 'r', encoding="utf-8") as file:
        cur_entry = Entry()
        indistinguishable = False
        last_change_border = 1000
        for line in file:
            if 'Color(h:' in line:
                values = [int(re.sub('\D', '', slice)) for slice in line.split(':')[2:]]
                cur_entry.colors.append(Color(values[0], values[1], values[2]))
            elif 'Target Index' in line:
                cur_entry.target_index = int(line.split(':')[1])
            elif 'Input:' in line:
                cur_entry.input = line.split(' ')[1].strip()
            elif 'Indistinguishable' in line:
                indistinguishable = True
            elif 'Context Prediction Correct: True' in line:
                cur_entry.correct_pred = True
            elif '---' in line:
                if not indistinguishable:
                    if cur_entry.change_borders == 0:
                        cur_entry.correct_pred = True
                    entries.append(cur_entry)
                else:
                    indistinguishable = False
                cur_entry = Entry()
            elif '===' in line:
                break

            last_change_border += 1
            if 'change_borders' in line:
                last_change_border = 0
            elif last_change_border == 2 or last_change_border == 3:
                cur_entry.change_borders += float(line.split(' ')[-1])
            elif 'Border Change:' in line:
                cur_entry.change_borders = float(line.split(':')[1].strip())
    
    return entries

def determine_errors(log_id):
    errors = []
    with open(f'colorful/logs/log{log_id}.log', 'r', encoding="utf-8") as file:
        failed_adaption = False
        indistinguishable = False
        for line in file:
            if 'Indistinguishable' in line:
                indistinguishable = True
            elif 'Model not changed' in line:
                failed_adaption = True
            elif '---' in line:
                if not indistinguishable:
                    errors.append(failed_adaption)
                    failed_adaption = False
                else:
                    indistinguishable = False
            elif '===' in line:
                break

    return errors