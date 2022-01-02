import setting
from file_io.sim_format import save_sample
from simulate_data.simulate import simulate

if __name__ == '__main__':
    bulk, frac, profile = simulate(setting.sc_label, setting.sc_value, setting.bulk_data, 5000, 5000)
    save_sample('2021-11-10', setting.sim_data_dir, bulk, frac, profile)