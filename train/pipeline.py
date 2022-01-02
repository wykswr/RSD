import setting
from file_io import load_data
from train.final_tune import *
from train.train_dae import *
from train.train_frac import *

# GPU memory limit
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def train_model(timestamp, cell_type, epoch, base_lr, bath_size, bulk_train, genes):
    gene_num = len(genes)
    cell_sep_train, _, cell_frac_train, _ = load_data.load_by_type(cell_type)
    base_save_dir = os.path.join(setting.train_result_dir, timestamp, cell_type)
    print('------------------------------------------------------------------------')
    print('Training {}...'.format(cell_type))
    print('------------------------------------------------------------------------')
    print('Training for dae ...')
    train_dae(base_save_dir, bulk_train, cell_sep_train, genes, gene_num, epoch, base_lr, bath_size)
    print('------------------------------------------------------------------------')
    print('Training for fraction ...')
    train_frac(base_save_dir, bulk_train, cell_frac_train, gene_num, epoch, base_lr, bath_size)
    print('------------------------------------------------------------------------')
    print('Final tune ...')
    final_tune(base_save_dir, bulk_train, cell_sep_train, cell_frac_train, genes, gene_num, epoch, base_lr/10, bath_size)


