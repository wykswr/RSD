import sys

import pandas as pd

from file_io import read_sim
from seaborn import heatmap

sys.path.append('..')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    dom = np.linalg.norm(a) * np.linalg.norm(b)
    return a.dot(b) / dom


if __name__ == '__main__':
    # label, pre = map(lambda x: normalize(read_sim(x), axis=1),
    #                  (r'D:\data\RSD\test\sc_norm.sim', r'D:\data\RSD\test\sc_pre.sim'))
    # heatmap(label, cmap="OrRd")
    # plt.show()
    # heatmap(pre, cmap="OrRd")
    # plt.show()
    # print(mean_squared_error(label, pre))
    # print(mean_absolute_error(label, pre))
    # for a, b in zip(label, pre):
    #     print(cos(a, b))
    genes = pd.read_table(r'D:\data\RSD\test\ori_sig.txt', index_col=0).index
    genes = set(read_sim(r'D:\data\RSD\test\bulk.sim').columns).intersection(genes)
    sc0 = normalize(read_sim(r'D:\data\RSD\test\bulk.sim')[genes], axis=1)
    sc1 = normalize(read_sim(r'D:\data\RSD\test\sc.sim')[genes], axis=1)
    sc2 = normalize(read_sim(r'D:\data\RSD\test\sc_pre.sim')[genes], axis=1)
    heatmap(sc0, cbar=False, cmap="hot")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    heatmap(sc1, cbar=False, cmap="hot")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    heatmap(sc2, cbar=False, cmap="hot")
    plt.xticks([])
    plt.yticks([])
    plt.show()

