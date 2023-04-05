import os

import pandas as pd

from others import setting


def read_sc_value(file: str) -> pd.DataFrame:
    name, suffix = os.path.splitext(file)
    if suffix == '.h5':
        tb = pd.read_hdf(file, 'df')
    else:
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


def read_sim(file: str) -> pd.DataFrame:
    return pd.read_table(file)


def save_sim(result: pd.DataFrame, path: str):
    result.to_csv(path, sep='\t', index=False)


def save_sample(name: str, bulk: pd.DataFrame, sc: pd.DataFrame):
    new_path = os.path.join(setting.save_dir, name)
    os.makedirs(new_path, exist_ok=True)
    save_sim(bulk, os.path.join(new_path, 'bulk.sim'))
    save_sim(sc, os.path.join(new_path, 'sc.sim'))
