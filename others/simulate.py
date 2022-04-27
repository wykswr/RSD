import datetime
import random
import sys

import pandas as pd
from file_io import save_sample
from others import setting
from Simulator import Simulator
sys.path.append('..')


def simulate_data(num: int, tag_cell: str, sc_label: str, sc_value: str, bulk_value: str) -> tuple:
    sim = Simulator(sc_label, sc_value, bulk_value)
    bulk, sc = list(), list()
    for i in range(num):
        this_bulk, this_sc = sim.simulate(random.randint(300, 800), tag_cell)
        bulk.append(this_bulk)
        sc.append(this_sc)
    genes = sim.get_genes()
    return pd.DataFrame(data=bulk, columns=genes), pd.DataFrame(data=sc, columns=genes)


def gen_record(num: int, aim_cell: str):
    timestamp = datetime.datetime.now()
    time_mark = '-'.join([str(timestamp.date()), str(timestamp.hour), str(timestamp.minute)])
    bk, sc = simulate_data(num, aim_cell, setting.sc_label, setting.sc_value, setting.bulk_value)
    save_sample(time_mark, bk, sc)


if __name__ == '__main__':
    gen_record(3000, 'CD8T')