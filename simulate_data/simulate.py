import numpy as np
import setting
from file_io.sim_format import *
import random


def gen_cell_frac(type_num: int, drop_rate: float) -> np.ndarray:
    assert 0 <= drop_rate < 1
    portion = np.random.random(type_num)
    mask = np.random.random(type_num) < drop_rate
    portion[mask] = 0
    total = portion.sum()
    if total == 0:
        return np.zeros(type_num)
    return portion / total


def gen_cell_num(cell_type: list, number: int, drop_rate: float) -> dict:
    fraction = gen_cell_frac(len(cell_type), drop_rate)
    dc = dict()
    remains = number
    for i in range(len(cell_type) - 1):
        cell_cnt = int(number * fraction[i])
        remains -= cell_cnt
        dc[cell_type[i]] = cell_cnt
    dc[cell_type[-1]] = remains
    return dc


def extract_label(scl: pd.DataFrame, cell_num: dict) -> list:
    labels = list()
    oid = random.choice(scl['orig.ident'].unique())
    for k, v in cell_num.items():
        labels.append(list(scl.loc[(scl.CellType == k) & (scl['orig.ident'] == oid)].sample(n=v, replace=True).index))
    return labels


def gen_bulk(labels: list, sc_value: pd.DataFrame) -> np.ndarray:
    if len(labels) == 0:
        return np.zeros(sc_value.shape[1])
    if type(labels[0]) is list:
        labels = sum(labels, [])
    mat = sc_value.loc[labels, :].values
    return mat.sum(axis=0) / mat.shape[0]


def simulate(sc_label: str, sc_value: str, bk_value: str, n_record: int, n_cell: int, drop_rate=0.05) -> tuple:
    genes = list(set(read_genes(sc_value)).intersection(set(read_genes(bk_value))))
    scv_tb = read_sc_value(sc_value)
    scl_tb = read_sc_label(sc_label)
    hq_genes = list()
    for i in genes:
        if scv_tb[i].var() > 0.1:
            hq_genes.append(i)
    genes = hq_genes
    scv_tb = scv_tb[genes]
    cell_type = list(scl_tb.CellType.unique())
    sim_bulks, sim_frac = list(), list()
    sim_profile = dict()
    for k in cell_type:
        sim_profile[k] = list()
    for i in range(n_record):
        cell_num = gen_cell_num(cell_type, n_cell, drop_rate)
        sim_frac.append(cell_num.values())
        labels = extract_label(scl_tb, cell_num)
        sim_bulks.append(gen_bulk(labels, scv_tb))
        for j in range(len(cell_type)):
            sim_profile.get(cell_type[j]).append(gen_bulk(labels[j], scv_tb))
    for k, v in sim_profile.items():
        sim_profile[k] = pd.DataFrame(v, columns=genes)
    return pd.DataFrame(sim_bulks, columns=genes), pd.DataFrame(sim_frac, columns=cell_type), sim_profile


if __name__ == '__main__':
    bulk, frac, profile = simulate(setting.sc_label, setting.sc_value, setting.bulk_data, 5000, 5000)
    save_sample('2021-11-10', setting.sim_data_dir, bulk, frac, profile)
