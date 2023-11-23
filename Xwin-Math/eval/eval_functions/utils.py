# ---------------------------------------------------------
# Xwin-Math
# Copyright (c) 2023 Xwin-Math Team
# Licensed under The MIT License [see LICENSE for details]
# Written by Weiqi Wang
# ---------------------------------------------------------

import os 
import json
import threading
import pandas as pd


class counter:
    def __init__(self) -> None:
        self.data_all = {}
        self.data_correct = {}

    def _add_if_none(self, type, subtype):
        for data in [self.data_correct, self.data_all]:
            if type not in data:
                data[type] = {}
            if subtype not in data[type]:
                data[type][subtype] = [0 for _ in range(10)]

    def add(self, type, subtype, level, correctness):
        self._add_if_none(type, subtype)
        self.data_all[type][subtype][level] += 1
        if correctness is True:
            self.data_correct[type][subtype][level] += 1

    @property
    def table(self):
        self.dataframe = pd.DataFrame(columns=["Type", "Subtype", "Level", "Correct", "Incorrect", "Total", "Accuracy"])
        for all_type, correct_type in zip(self.data_all.items(), self.data_correct.items()):
            for all_subtype, correct_subtype in zip(all_type[1].items(), correct_type[1].items()):
                for index, (all_level, correct_level) in enumerate(zip(all_subtype[1], correct_subtype[1])):
                    if all_level != 0:
                        self.dataframe.loc[len(self.dataframe)] = {
                            "Type": all_type[0],
                            "Subtype": all_subtype[0],
                            "Level": str(index),
                            "Correct": correct_level,
                            "Incorrect": all_level - correct_level,
                            "Total": all_level,
                            "Accuracy": correct_level/all_level
                        }
                all_total_level = sum(all_subtype[1])
                correct_total_level = sum(correct_subtype[1])
                if all_total_level != 0 and len(all_subtype[1]) - all_subtype[1].count(0) != 1:
                    self.dataframe.loc[len(self.dataframe)] = {
                        "Type": all_type[0],
                        "Subtype": all_subtype[0],
                        "Level": "TOTAL",
                        "Correct": correct_total_level,
                        "Incorrect": all_total_level - correct_total_level,
                        "Total": all_total_level,
                        "Accuracy": correct_total_level/all_total_level
                    }

            all_total_subtype = sum([sum(lst) for lst in all_type[1].values()])
            correct_total_subtype = sum([sum(lst) for lst in correct_type[1].values()])
            if all_total_subtype != 0 and len(all_type[1]) != 1:
                self.dataframe.loc[len(self.dataframe)] = {
                    "Type": all_type[0],
                    "Subtype": "TOTAL",
                    "Level": "TOTAL",
                    "Correct": correct_total_subtype,
                    "Incorrect": all_total_subtype - correct_total_subtype,
                    "Total": all_total_subtype,
                    "Accuracy": correct_total_subtype/all_total_subtype
                }
        return self.dataframe


class TimeoutThread:  
    def __init__(self, function, args=(), timeout=3):
        self.timeout = timeout
        self.function = function
        self.args = args
        self.result = None
        self.finished = threading.Event()

    def run(self):
        self.thread = threading.Thread(target=self.call_function)
        self.thread.start()
        self.thread.join(timeout=self.timeout)

        if self.thread.is_alive():
            return None
        else:
            return self.result

    def call_function(self):
        self.result = self.function(*self.args)
        self.finished.set()


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def add_prefix_to_filename(path, prefix):  
    dirname, filename = os.path.split(path)
    new_filename = prefix + "_" + filename
    new_path = os.path.join(dirname, new_filename)
    return new_path