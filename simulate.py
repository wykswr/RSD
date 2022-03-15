import datetime
import random
import pandas as pd
from file_io.sim_io import save_sample
from setting import *
from simulate.Simulator import Simulator


def gen_record(num: int, aim_cell: str):
    sim = Simulator(sc_label, sc_value, bulk_data)
    bulk, sc = list(), list()
    for i in range(num):
        this_bulk, this_sc = sim.simulate(random.randint(300, 2000), aim_cell)
        bulk.append(this_bulk)
        sc.append(this_sc)
    genes = sim.get_genes()
    timestamp = datetime.datetime.now()
    time_mark = '-'.join([str(timestamp.date()), str(timestamp.hour), str(timestamp.minute)])
    save_sample(time_mark, pd.DataFrame(data=bulk, columns=genes), pd.DataFrame(data=sc, columns=genes))


if __name__ == '__main__':
    gen_record(20, 'CD8T')