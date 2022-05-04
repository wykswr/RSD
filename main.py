import os
import random
import sys
from argparse import ArgumentParser
import pandas as pd
from sklearn.preprocessing import normalize
from PdSimulator import PdSimulator, DbSimulator
from file_io import read_sc_value, save_sim, read_sim


def get_args():
    parser = ArgumentParser(description='RSD is a deep-learning tool to predict the cell-type specific genes profile')
    parser.add_argument('input', help='bulk RNA-seq file')
    parser.add_argument('--singleCellValue', '-v', help='the single cell value of reference database')
    parser.add_argument('--singleCellLabel', '-l', help='the annotation of single cell value file')
    parser.add_argument('--targetCell', '-c', help='the target cell type of deconvolution')
    parser.add_argument('--output', '-o', help='the path to save the result file')
    parser.add_argument('--sim', '-s', action='store_true', help='whether use the exist sim bulk and sc')
    parser.add_argument('--database', '-d', action='store_true', help='whether use sqlite3 as singleCellValue '
                                                                      'and singleCellLabel')
    parser.add_argument('--sqlite', '-q', help='the sqlite3 database')
    parser.add_argument('--bulk', '-b', help='the simulated bulk RNA-seq')
    parser.add_argument('--profile', '-p', help='the simulated cell-type specific profile')
    parser.add_argument('--generateSim', '-g', action='store_true', help='whether generate the sim file only')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.sim:
        sim_bulk, sim_sc = map(read_sim, (args.bulk, args.profile))
    else:
        if args.database:
            cook = DbSimulator(args.sqlite, args.input)
        else:
            cook = PdSimulator(args.singleCellLabel, args.singleCellValue, args.input)
        sim_bulk, sim_sc = list(), list()
        for i in range(3000):
            this_bulk, this_sc = cook.simulate(random.randint(300, 800), args.targetCell)
            sim_bulk.append(this_bulk)
            sim_sc.append(this_sc)
        genes = cook.get_genes()
        if args.database:
            cook.close()
        del cook
        sim_bulk, sim_sc = pd.DataFrame(data=sim_bulk, columns=genes), pd.DataFrame(data=sim_sc, columns=genes)
        sim_bulk, sim_sc = map(lambda x: normalize(x, axis=1), (sim_bulk, sim_sc))
    if args.generateSim:
        save_sim(sim_bulk, os.path.join(args.output, 'bulk.sim'))
        save_sim(sim_sc, os.path.join(args.output, '{}.sim'.format(args.targetCell)))
        sys.exit(0)
    from train import train
    from DAE import FC1
    ori_bulk = read_sc_value(args.input)
    genes = set(ori_bulk.columns).intersection(sim_bulk.columns)
    ori_bulk, sim_bulk, sim_sc = map(lambda x: x[genes], (ori_bulk, sim_bulk, sim_sc))
    mod = FC1(gene_num=len(genes))
    train(mod, sim_bulk, sim_sc)
    del sim_bulk, sim_sc
    samples = ori_bulk.index
    result = pd.DataFrame(normalize(mod.predict(normalize(ori_bulk, axis=1)), axis=1), columns=genes, index=samples)
    save_sim(result, os.path.join(args.output, '{}_predicted.sim'.format(args.targetCell)))
