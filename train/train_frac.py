import os

from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from file_io.write_result import *
from structure.FracFC import *


def train_frac(base_save_dir, RNA_seq_train, cell_frac_train, gene_num, epoch, lr, batch_size):
    kfold = KFold(n_splits=4, shuffle=True)
    fold_no = 1
    best_loss = float('inf')
    for train_index, val_index in kfold.split(X=RNA_seq_train, y=cell_frac_train):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        train_data, train_label = RNA_seq_train[train_index,], cell_frac_train[train_index,]
        val_data, val_label = RNA_seq_train[val_index,], cell_frac_train[val_index]

        save_dir = os.path.join(base_save_dir, 'frac', str(fold_no), '')
        os.makedirs(save_dir, exist_ok=True)
        dae_model = DaeFC(gene_num=gene_num)
        dae_model.load_weights(os.path.join(base_save_dir, 'dae', 'best_model', ''))
        dae_model.trainable = True

        f_model = FracFC(dae=dae_model)
        METRICS = [
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae")
        ]

        f_model.compile(
            loss="mae",
            optimizer=Adam(learning_rate=lr),
            metrics=METRICS
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=save_dir,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_freq='epoch',
            save_best_only=True)

        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0.01,
                                  mode='min',
                                  patience=20,
                                  restore_best_weights=True)

        history = f_model.fit(train_data,
                              train_label,
                              batch_size=batch_size,
                              epochs=epoch,
                              verbose=1,
                              shuffle=True,
                              callbacks=[earlystop, model_checkpoint_callback],
                              validation_data=(val_data,
                                               val_label))
        print('validation ')
        f_model.load_weights(save_dir)
        val_results = f_model.evaluate(val_data, val_label, verbose=1)
        for name, value in zip(f_model.metrics_names, val_results):
            print(name, ': ', value)
        val_loss = history.history['val_loss']
        write_list(os.path.join(save_dir, 'log.txt'), val_loss)
        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            f_model.save_weights(os.path.join(base_save_dir, 'frac', 'best_model', ''))
        write_result(f_model, save_dir, val_data, train_data, val_label, gene_list=None, train_label_frac=train_label,
                     val_label_frac=val_label, model_type="frac")
        fold_no += 1
