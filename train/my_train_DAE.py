import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError

from file_io.write_result import write_list, write_result
from structure.DaeFC import *


def train_dae(base_save_dir, RNA_seq_train, cell_sep_train, gene_list, gene_num, epoch, lr, batch_size):
    train_data, val_data, train_label, val_label = train_test_split(RNA_seq_train, cell_sep_train, test_size=0.1)
    save_dir = os.path.join(base_save_dir, 'dae', '')
    os.makedirs(save_dir, exist_ok=True)
    dae_model = DaeFC(gene_num=gene_num)
    metric = [MeanSquaredError(name="mse"), MeanAbsoluteError(name="mae")]
    dae_model.compile(loss="mse", optimizer=Adam(learning_rate=lr), metrics=metric)
    model_checkpoint = ModelCheckpoint(filepath=save_dir,
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
    history = dae_model.fit(train_data,
                            train_label,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            shuffle=True,
                            callbacks=[earlystop, model_checkpoint],
                            validation_data=(val_data, val_label))
    dae_model.load_weights(save_dir)
    val_loss = history.history['val_loss']
    write_list(os.path.join(save_dir, 'log.txt'), val_loss)
    write_result(dae_model, save_dir, val_data, train_data, val_label, gene_list, model_type="dae")
