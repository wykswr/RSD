import getopt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_vec(file) -> np.ndarray:
    with open(file) as src:
        vec = []
        for l in src:
            vec.append(float(l))
        return np.array(vec)


log = 'log.txt'
label_frac = 'train_frac_label.txt'
pre_frac = 'train_frac_pred.txt'

opts, args = getopt.getopt(sys.argv[1:], 'i:prl')
for k, v in opts:
    if k in ('-i',):
        ls = v

plt.style.use('ggplot')

for k, v in opts:
    if k in ('-p',):
        vec_label, vec_pre = map(lambda x: read_vec(os.path.join(ls, x)), (label_frac, pre_frac))
        plt.plot([0, 1], [0, 1])
        plt.plot(vec_label, vec_pre, '.')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('label')
        plt.ylabel('pre')
        plt.show()
    if k in ('-r',):
        vec_label, vec_pre = map(lambda x: read_vec(os.path.join(ls, x)), (label_frac, pre_frac))
        vec_res = vec_pre - vec_label
        plt.plot(vec_res, vec_pre, '.')
        plt.xlabel('res')
        plt.ylabel('pre')
        plt.show()
    if k in ('-l',):
        vec_log = read_vec(os.path.join(ls, log))
        plt.plot(vec_log)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
