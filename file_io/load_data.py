import json

import numpy as np
import pandas as pd

import setting
from .scan_dir import scan


def get_index():
    with open(scan(setting.sim_data_dir)['index']) as handle:
        return json.load(handle)


def load_genes_bulk(index=get_index()):
    bulk_seq = scan(setting.sim_data_dir)['bulk']
    bulk_table = pd.read_table(bulk_seq).fillna(0)
    genes = bulk_table.columns
    train_bulk = bulk_table.iloc[index].values.astype(np.float64)
    test_bulk = bulk_table.drop(index).values.astype(np.float64)
    return genes, train_bulk, test_bulk


def load_by_type(c_type, index=get_index()):
    file_dc = scan(setting.sim_data_dir)
    cell_frac = file_dc['frac']
    cell_mat = file_dc[c_type]
    seq_table, frac_table = map(lambda x: pd.read_table(x).fillna(0), (cell_mat, cell_frac))
    frac_table = frac_table[c_type]
    train_seq, train_frac = map(lambda x: x.iloc[index].values.astype(np.float64), (seq_table, frac_table))
    test_seq, test_frac = map(lambda x: x.drop(index).values.astype(np.float64), (seq_table, frac_table))
    train_frac, test_frac = map(lambda x: np.vstack((x, 1 - x)).T, (train_frac, test_frac))
    return train_seq, test_seq, train_frac, test_frac
