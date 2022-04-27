import random
from argparse import ArgumentParser

import pandas as pd
from sklearn.preprocessing import normalize

from DAE import FC1
from Simulator import Simulator
from file_io import read_sc_value, save_sim
from train import train


def predict(sc_value: str, sc_label: str, bulk: str, tag_cell: str, out_dir: str):
    cook = Simulator(sc_label, sc_value, bulk)
    sim_bulk, sim_sc = list(), list()
    for i in range(2000):
        this_bulk, this_sc = cook.simulate(random.randint(300, 800), tag_cell)
        sim_bulk.append(this_bulk)
        sim_sc.append(this_sc)
    genes = cook.get_genes()
    sim_bulk, sim_sc = pd.DataFrame(data=sim_bulk, columns=genes), pd.DataFrame(data=sim_sc, columns=genes)
    ori_bulk = read_sc_value(bulk)[genes]
    samples = ori_bulk.index
    mod = FC1(gene_num=len(genes))
    sim_bulk, sim_sc = map(lambda x: normalize(x, axis=1), (sim_bulk, sim_sc))
    train(mod, sim_bulk, sim_sc)
    result = pd.DataFrame(normalize(mod.predict(normalize(ori_bulk, axis=1)), axis=1), columns=genes, index=samples)
    save_sim(result, path=out_dir)


def get_args():
    ap = ArgumentParser()
    ap.add_argument('input', help='bulk RNA-seq file')
    ap.add_argument('--singleCellValue', '-v', help='the single cell value of reference database')
    ap.add_argument('--singleCellLabel', '-l', help='the annotation of single cell value file')
    ap.add_argument('--output', '-o', help='the full path name of the result file')
    return ap.parse_args()


if __name__ == '__main__':
    args = get_args()
    predict(args.singleCellValue, args.singleCellLabel, args.input, 'CD8T', args.output)

