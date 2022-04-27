import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from DAE import FC1
from sklearn.model_selection import train_test_split
from file_io import read_sim, save_sim, read_sc_value
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, CosineSimilarity
from sklearn.preprocessing import normalize


def train(mod, X, y, lr, batch_size, loss):
    if loss == 'cos':
        loss = keras.losses.CosineSimilarity(axis=1)
    epoch = 100
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    mtc = [MeanSquaredError(name="mse"),
           MeanAbsoluteError(name="mae"),
           CosineSimilarity(name='cos', axis=1)]
    mod.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=mtc)
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0.0001,
                       mode='min',
                       patience=10,
                       restore_best_weights=True)
    return mod.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,
                   verbose=2,
                   shuffle=True,
                   callbacks=[es, ],
                   validation_data=(X_val, y_val))


if __name__ == '__main__':
    bk, sc = map(read_sim, (r'D:\data\RSD\independent\bulk.sim', r'D:\data\RSD\independent\sc.sim'))
    genes = bk.columns
    bk, sc = map(lambda x: normalize(x, axis=1), (bk, sc))
    gn = bk.shape[1]
    model = FC1(gene_num=gn)
    history = train(model, bk, sc, lr=0.005, batch_size=32, loss='cos')
    # print('evaluate on test:')
    # test_bulk, test_sc = map(lambda x: normalize(read_sim(x), axis=1),
    #                          (r'D:\data\RSD\test\bulk.sim', r'D:\data\RSD\test\sc.sim'))
    # model.evaluate(test_bulk, test_sc, verbose=2)
    tag = np.log(1+read_sc_value(r'D:\data\RSD\independent\id_bulk.txt')[genes].values)
    test_pre = model.predict(normalize(tag, axis=1))
    save_sim(pd.DataFrame(test_pre, columns=genes), r'D:\data\RSD\independent\sc_pre.sim')




    # record = history.history
    # record['loss'] = 'cos'
    # record['learning_rate'] = 0.005
    # record['batch_size'] = 32
    # with open(r'D:\data\RSD\result\train3.json', 'w') as fp:
    #     json.dump(record, fp)
