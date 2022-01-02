import numpy as np


def write_result(model, save_dir, val_data, train_data, val_label, gene_list, train_label_frac=None,
                 val_label_frac=None, model_type=None):
    if model_type == "final":
        val_mat_out, val_frac_out = model.predict(val_data)
        train_mat_out, train_frac_out = model.predict(train_data)
    elif model_type == "frac":
        val_frac_out = model.predict(val_data)
        train_frac_out = model.predict(train_data)
    elif model_type == "dae":
        val_mat_out = model.predict(val_data)

    if model_type == "frac" or model_type == "final":
        ## save cell fraction
        ypred_file = save_dir + "train_frac_pred.txt"
        write_matrix(ypred_file, train_frac_out[:, 0].reshape(train_frac_out.shape[0], 1))

        label_file = save_dir + "train_frac_label.txt"
        write_matrix(label_file, train_label_frac[:, 0].reshape(train_label_frac.shape[0], 1))

        ypred_file = save_dir + "val_frac_pred.txt"
        write_matrix(ypred_file, val_frac_out[:, 0].reshape(val_frac_out.shape[0], 1))

        label_file = save_dir + "val_frac_label.txt"
        write_matrix(label_file, val_label_frac[:, 0].reshape(val_label_frac.shape[0], 1))
    if model_type == "dae" or model_type == "final":
        ## save cell sepecific matrix
        ypred_file = save_dir + "val_mat_pred.txt"
        write_matrix(ypred_file, val_mat_out)

        label_file = save_dir + "val_mat_label.txt"
        write_matrix(label_file, val_label)

        ## save gene ordered by expression level
        label_gene_sort_file = save_dir + "label_gene_sort.txt"
        write_gene_sort(label_gene_sort_file, gene_list, val_label)
        ypred_gene_sort_file = save_dir + "ypred_gene_sort.txt"
        write_gene_sort(ypred_gene_sort_file, gene_list, val_mat_out)


def write_matrix(PATH, matrix):
    fw = open(PATH, 'w')
    for r in range(matrix.shape[0]):
        tmp_row = matrix[r].tolist()
        tmp_row = [str(int) for int in tmp_row]
        str_of_row = " ".join(tmp_row)
        fw.write(str_of_row + '\n')
    fw.close()


def write_list(PATH, list):
    f = open(PATH, 'w')
    for ele in list:
        f.write(str(ele) + '\n')
    f.close()


def write_gene_sort(PATH, gene_list, data):
    express_sum = np.sum(data, axis=0)
    gene_list_sort = [gene_list[x] for x in express_sum.argsort()]
    write_list(PATH, gene_list_sort)
