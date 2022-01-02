import time
from file_io import load_data, scan_dir
from train import pipeline


if __name__ == '__main__':
    # begin parameters
    epoch = 1000
    base_lr = 0.0001
    batch_size = 32
    # end parameters
    genes, bulk_train, bulk_test = load_data.load_genes_bulk()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    cells = scan_dir.get_cells()
    for cell in cells:
        pipeline.train_model(timestamp, cell, epoch, base_lr, batch_size, bulk_train, genes)
