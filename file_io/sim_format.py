import pandas as pd
import os
from itertools import combinations


def read_sc_value(file: str) -> pd.DataFrame:
    tb = pd.read_table(file)
    tb.set_index('GeneSymbol', inplace=True)
    return tb.transpose()


def read_sc_label(file: str) -> pd.DataFrame:
    tb = pd.read_table(file)
    tb.set_index('cellBarcode', inplace=True)
    return tb


def read_genes(file: str) -> list:
    genes = list()
    with open(file) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('geneSymbol'):
                continue
            genes.append(line.split()[0])
    return genes


def save_sim(result: pd.DataFrame, path: str):
    result.to_csv(path, sep='\t', index=False)


def save_sample(name: str, path: str, bulk: pd.DataFrame, frac: pd.DataFrame, profile: dict):
    new_path = os.path.join(path, name)
    os.makedirs(new_path, exist_ok=True)
    save_sim(bulk, os.path.join(new_path, 'bulk.sim'))
    save_sim(frac, os.path.join(new_path, 'frac.sim'))
    for k, v in profile.items():
        save_sim(v, os.path.join(new_path, k+'.sim'))
