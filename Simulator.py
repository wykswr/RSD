import random
import numpy as np
from file_io import read_sc_label, read_genes, read_sc_value
import sqlite3


class PdSimulator:
    def __init__(self, sc_label: str, sc_value: str, bk_value: str):
        self.scv_tb = read_sc_value(sc_value)
        self.genes = list(set(self.scv_tb.columns).intersection(set(read_genes(bk_value))))
        self.genes = sorted(self.genes)
        self.scv_tb = self.scv_tb[self.genes]
        self.scl_tb = read_sc_label(sc_label)
        self.drop_rate = 0.05
        self.cell_type = list(self.scl_tb.CellType.unique())
        self.cell_type = sorted(self.cell_type)

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
            v = int(v)
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
        n_cell_dc = self.gen_cell_num(n_cell)
        labels = self.extract_label(n_cell_dc)
        type_dc = dict(zip(self.cell_type, labels))
        bulk = self.gen_bulk(labels)
        sc = self.gen_bulk(type_dc[aim_cell])
        return bulk, sc


class DbSimulator:
    def __init__(self, merge: str, bk_value: str):
        self.conn = sqlite3.connect(merge)
        self.cur = self.conn.cursor()
        self.cur.execute('SELECT DISTINCT gene FROM genes;')
        db_genes = np.array([x[0] for x in self.cur.fetchall()])
        self.genes = sorted(list(set(db_genes).intersection(set(read_genes(bk_value)))))
        idxes = np.array(range(len(db_genes)))
        self.idx = [idxes[db_genes == x][0] for x in self.genes]
        self.cur.execute('SELECT DISTINCT cellType FROM annotation;')
        self.cell_type = [x[0] for x in self.cur.fetchall()]
        self.cur.execute('SELECT DISTINCT origIdent FROM annotation;')
        self.orig_ident = [x[0] for x in self.cur.fetchall()]
        self.drop_rate = 0.05

    def close(self):
        self.cur.close()
        self.conn.close()

    def get_genes(self) -> list:
        return self.genes

    def analyze(self, buff: str) -> np.ndarray:
        return np.frombuffer(buff)[self.idx]

    def get_profile(self, samples: list) -> np.ndarray:
        result = list()
        for s in samples:
            self.cur.execute('SELECT profile FROM single_cell WHERE sample = ?;', (s,))
            result.append(self.analyze(self.cur.fetchone()[0]))
        return np.array(result)

    def get_sample(self, cell_type: str, orig_ident: str) -> list:
        self.cur.execute('SELECT sample FROM annotation WHERE cellType=? AND origIdent=?;', (cell_type, orig_ident))
        return [x[0] for x in self.cur.fetchall()]

    def set_drop_rate(self, drop_rate: float):
        assert 0 <= drop_rate <= 1
        self.drop_rate = drop_rate

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
        labels = list()
        oid = random.choice(self.orig_ident)
        for k, v in cell_num.items():
            v = int(v)
            labels.append(
                random.choices(self.get_sample(k, oid), k=v)
            )
        return labels

    def gen_bulk(self, labels: list) -> np.ndarray:
        if len(labels) == 0:
            return np.zeros(len(self.genes))
        if type(labels[0]) is list:
            labels = sum(labels, [])
        mat = self.get_profile(labels)
        return mat.sum(axis=0) / mat.shape[0]

    def simulate(self, n_cell: int, aim_cell: str):
        n_cell_dc = self.gen_cell_num(n_cell)
        labels = self.extract_label(n_cell_dc)
        type_dc = dict(zip(self.cell_type, labels))
        bulk = self.gen_bulk(labels)
        sc = self.gen_bulk(type_dc[aim_cell])
        return bulk, sc
