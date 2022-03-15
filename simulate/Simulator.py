import random

import numpy as np

from file_io.sim_io import read_sc_label, read_genes, read_sc_value


class Simulator:
    def __init__(self, sc_label: str, sc_value: str, bk_value: str):
        self.scv_tb = read_sc_value(sc_value)
        self.genes = list(set(self.scv_tb.columns).intersection(set(read_genes(bk_value))))
        self.scv_tb = read_sc_value(sc_value)[self.genes]
        self.scl_tb = read_sc_label(sc_label)
        self.drop_rate = 0.05
        self.cell_type = list(self.scl_tb.CellType.unique())

    def set_drop_rate(self, drop_rate: float):
        assert 0 <= drop_rate <= 1
        self.drop_rate = drop_rate

    def get_genes(self) -> list:
        return self.genes

    def gen_cell_frac(self) -> np.ndarray:
        type_num = len(self.cell_type)
        portion = np.random.random(type_num)
        mask = np.random.random(type_num) < self.drop_rate
        portion[mask] = 0
        total = portion.sum()
        if total == 0:
            return np.zeros(type_num)
        return portion / total

    def gen_cell_num(self, number: int) -> dict:
        fraction = self.gen_cell_frac()
        dc = dict()
        remains = number
        for i in range(len(self.cell_type) - 1):
            cell_cnt = int(number * fraction[i])
            remains -= cell_cnt
            dc[self.cell_type[i]] = cell_cnt
        dc[self.cell_type[-1]] = remains
        return dc

    def extract_label(self, cell_num: dict) -> list:
        scl = self.scl_tb
        labels = list()
        oid = random.choice(scl['orig.ident'].unique())
        for k, v in cell_num.items():
            labels.append(
                list(scl.loc[(scl.CellType == k) & (scl['orig.ident'] == oid)].sample(n=v, replace=True).index))
        return labels

    def gen_bulk(self, labels: list) -> np.ndarray:
        if len(labels) == 0:
            return np.zeros(self.scv_tb.shape[1])
        if type(labels[0]) is list:
            labels = sum(labels, [])
        mat = self.scv_tb.loc[labels].values
        return mat.sum(axis=0) / mat.shape[0]

    def simulate(self, n_cell: int, aim_cell: str):
        cell_num = self.gen_cell_num(n_cell)
        labels = self.extract_label(cell_num)
        type_dc = dict(zip(self.cell_type, labels))
        bulk = self.gen_bulk(labels)
        sc = self.gen_bulk(type_dc[aim_cell])
        return bulk, sc
