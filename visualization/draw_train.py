import matplotlib.pyplot as plt
import json
import numpy as np


def draw(file: str):
    with open(file) as fp:
        dc = json.load(fp)
    dc['cos'] = -np.array(list(dc['cos']))
    dc['val_cos'] =-np.array(list(dc['val_cos']))
    plt.subplot(3, 1, 1)
    mt = ['mae', 'mse', 'cos']
    cnt = 1
    for k in mt:
        plt.subplot(len(mt), 1, cnt)
        if dc['loss'] == k:
            plt.plot(dc['val_'+k], 'r', label='val_'+k)
            # plt.plot(dc[k], 'r', label=k)
        else:
            plt.plot(dc['val_'+k], label='val_'+k)
            # plt.plot(dc[k], label=k)
        plt.legend()
        if cnt != len(mt):
            plt.xticks([])
        if cnt == len(mt):
            plt.xlabel('train epoch')
        cnt += 1
    plt.savefig(r'C:\Users\Wangyk\Desktop\grad_thesis\pictures\3metrics_val.pdf')
    plt.clf()


if __name__ == '__main__':
    draw(r'D:\data\RSD\result\train3.json')
