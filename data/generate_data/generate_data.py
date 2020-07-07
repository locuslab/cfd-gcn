import os
import pickle
import shutil
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

import torch

import SU2
from su2torch import SU2Module
from su2torch.su2_function_mpi import modify_config


REYNOLDS_LIST = [None]
AOA_TRAIN_LIST = [float(x) for x in range(-10, 10+1)]
# AOA_TRAIN_LIST = [1., 5., 10.]
AOA_TEST_LIST = []
MACH_TRAIN_LIST = [0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]
MACH_TEST_LIST = [0.25, 0.45, 0.65]


def main(config_filename, mesh_file, output_dir):
    """ From paper:
        Simulations are performed at Reynolds numbers 0.5, 1, 2, and 3 × 10 6 , respectively, and a low Mach
        number of 0.2 is selected to be representative of wind turbine conditions. At each Reynolds number, the
        simulation is performed for different airfoils with a sweep of angles of attack from α = 0 ◦to α = 20.
    """
    os.makedirs(output_dir / 'outputs_train', exist_ok=False)
    os.makedirs(output_dir / 'outputs_test', exist_ok=False)
    shutil.copy(mesh_file, output_dir / 'mesh_fine.su2')

    vx_sum = vy_sum = p_sum = vx2_sum = vy2_sum = p2_sum = 0
    vx_max = vy_max = p_max = -1e10
    vx_min = vy_min = p_min = 1e10
    n = 0
    for reynolds in tqdm(REYNOLDS_LIST, desc='Reynds', dynamic_ncols=True):
        for aoa in tqdm(AOA_TRAIN_LIST + AOA_TEST_LIST, desc='AOA   ', dynamic_ncols=True):
            for mach in tqdm(MACH_TRAIN_LIST + MACH_TEST_LIST, desc='Mach  ', dynamic_ncols=True):
                if reynolds is not None:
                    reynolds_config = {'REYNOLDS_NUMBER': reynolds}
                    shutil.copy(config_filename, 'reynolds_config_temp.cfg')
                    modify_config(SU2.io.Config(config_filename), reynolds_config, outfile='reynolds_config_temp.cfg')
                    config_filename = 'reynolds_config_temp.cfg'

                su2 = SU2Module(config_filename, mesh_file)
                outputs = su2(torch.tensor([aoa]).unsqueeze(0), torch.tensor([mach]).unsqueeze(0))
                outputs = [o.squeeze(0).numpy() for o in outputs]

                if aoa in AOA_TRAIN_LIST and mach in MACH_TRAIN_LIST:
                    vx_max = max(outputs[0].max(), vx_max)
                    vy_max = max(outputs[1].max(), vy_max)
                    p_max = max(outputs[2].max(), p_max)
                    vx_min = min(outputs[0].min(), vx_min)
                    vy_min = min(outputs[1].min(), vy_min)
                    p_min = min(outputs[2].min(), p_min)
                    vx_mean = outputs[0].mean()
                    vy_mean = outputs[1].mean()
                    p_mean = outputs[2].mean()
                    vx_sum += vx_mean
                    vy_sum += vy_mean
                    p_sum += p_mean
                    vx2_sum += vx_mean ** 2
                    vy2_sum += vy_mean ** 2
                    p2_sum += p_mean ** 2
                    n += 1

                    train_test_dir = 'outputs_train'
                else:
                    train_test_dir = 'outputs_test'
                path = output_dir / train_test_dir / 'outputs_re_{}_aoa_{}_mach_{}.pkl'.format(reynolds, aoa, mach)
                with open(path, 'wb') as f:
                    pickle.dump(outputs, f)

    data_means = [vx_sum / n, vy_sum / n, p_sum / n]
    data_stds = [np.sqrt(vx2_sum / n - data_means[0] ** 2),
                 np.sqrt(vy2_sum / n - data_means[1] ** 2),
                 np.sqrt(p2_sum / n - data_means[2] ** 2)]
    with open(output_dir / 'train_mean_std.pkl', 'wb') as f:
        pickle.dump([data_means, data_stds], f)
    data_max = [vx_max, vy_max, p_max]
    data_min = [vx_min, vy_min, p_min]
    with open(output_dir / 'train_max_min.pkl', 'wb') as f:
        pickle.dump([data_max, data_min], f)


if __name__ == '__main__':
    from su2torch import activate_su2_mpi
    activate_su2_mpi()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', required=True)
    parser.add_argument('--mesh-file', '-m', required=True)
    parser.add_argument('--output-dir', '-o', default='generated_data')
    args = vars(parser.parse_args())

    with torch.no_grad():
        main(args['config_file'], args['mesh_file'], Path(args['output_dir']))
