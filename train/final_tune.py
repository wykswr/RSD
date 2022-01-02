import os

from sklearn.model_selection import KFold
from tensorflow.keras.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam

from file_io.write_result import *
from structure.FracFC import *
from structure.final_model import *


def final_tune(base_save_dir, RNA_seq_train, cell_sep_train, cell_frac_train, gene_list, gene_num, epoch, lr, batch_size):
    kfold = KFold(n_splits=4, shuffle=True)
    min_delta = 0.01
    max_patience = 20
    best_loss = float('inf')
    fold_no = 1
    for train_index, val_index in kfold.split(X=RNA_seq_train, y=cell_sep_train):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        train_data, train_label, train_label_frac = RNA_seq_train[train_index,], cell_sep_train[train_index,], cell_frac_train[train_index, ]
        val_data, val_label, val_label_frac = RNA_seq_train[val_index,], cell_sep_train[val_index,], cell_frac_train[val_index, ]

        save_dir = os.path.join(base_save_dir, 'final_tune', str(fold_no), '')
        os.makedirs(save_dir, exist_ok=True)
        dae_model = DaeFC(gene_num=gene_num)
        dae_model.load_weights(os.path.join(base_save_dir, 'dae', 'best_model', ''))

        f_model = FracFC(dae=dae_model)
        f_model.load_weights(os.path.join(base_save_dir, 'frac', 'best_model', ''))

        final_model = FinalModel(dae=dae_model, frac=f_model)
        final_model_optimizer = Adam(learning_rate=lr)

        val_loss = []
        best_loss_fold = 10
        pre_loss = 10
        patience = 0
        for e in range(epoch):
            print('------------------------------------------------------------------------')
            print(f'Processing epoch {e} ..........')
            ## shuffle data
            train_data, train_label, train_label_frac = shuffle_data(train_data, train_label, train_label_frac)
            ## batch index
            batch_num = int(train_data.shape[0] // batch_size)
            batch_start = 0
            batch_end = batch_start + batch_size

            for b in range(batch_num):
                minibatch_data = train_data[batch_start:batch_end, ]
                minibatch_label = train_label[batch_start:batch_end, ]
                minibatch_label_frac = train_label_frac[batch_start:batch_end, ]

                with tf.GradientTape() as tape:
                    mat_out, frac_out = final_model(minibatch_data, training=True)
                    loss1 = mean_squared_error(y_true=minibatch_label, y_pred=mat_out)
                    loss2 = mean_absolute_error(y_true=minibatch_label_frac, y_pred=frac_out)
                    loss = loss1 + loss2
                gradients_of_final_model = tape.gradient(loss, final_model.trainable_variables)
                final_model_optimizer.apply_gradients(zip(gradients_of_final_model, final_model.trainable_variables))

                batch_start = batch_start + batch_size
                batch_end = batch_end + batch_size

            mat_out, frac_out = final_model(val_data)
            val_l1 = mean_squared_error(y_true=val_label, y_pred=mat_out)
            val_l2 = mean_absolute_error(y_true=val_label_frac, y_pred=frac_out)
            vl = val_l1 + val_l2
            val_loss.append(np.mean(vl))

            print(f'dae loss = {np.mean(val_l1)}  frac loss = {np.mean(val_l2)}..........')
            print(f'current loss = {np.mean(vl)}  best loss = {best_loss_fold}..........')
            print(f'patience = {patience}')

            cur_best_loss = min(val_loss)
            if cur_best_loss < best_loss_fold:
                best_loss_fold = cur_best_loss
                final_model.save_weights(save_dir)
            if cur_best_loss < best_loss:
                best_loss = cur_best_loss
                final_model.save_weights(os.path.join(base_save_dir, 'final_tune', 'best_model', ''))

            if abs(pre_loss - best_loss_fold) < min_delta:
                patience = patience + 1

            if patience == max_patience:
                print("Early stop .............................")
                break
            pre_loss = cur_best_loss

        write_list(os.path.join(save_dir, 'log.txt'), val_loss)
        final_model.load_weights(save_dir)

        write_result(final_model, save_dir, val_data, train_data, val_label, gene_list, train_label_frac,
                     val_label_frac, "final")

        fold_no = fold_no + 1


def shuffle_data(data, label, label_frac):
    sf_index = np.random.choice(data.shape[0], data.shape[0], replace=False)
    sf_data = data[sf_index,]
    sf_label = label[sf_index,]
    sf_label_frac = label_frac[sf_index,]
    return sf_data, sf_label, sf_label_frac