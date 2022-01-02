import os

import setting


def split_name(file: str) -> str:
    name, _ = os.path.splitext(file)
    return name


def scan(path: str) -> dict:
    dc = dict()
    for file in os.listdir(path):
        dc[split_name(file)] = os.path.join(path, file)
    return dc


def get_cells() -> list:
    cells = list()
    for c in scan(setting.sim_data_dir).keys():
        if c not in ['frac', 'index', 'bulk']:
            cells.append(c)
    return cells
