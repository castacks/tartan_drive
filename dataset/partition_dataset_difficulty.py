import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from os_util import maybe_mkdir

if __name__ == '__main__':
    """
    Use the metric in the paper (integrated dz) to partition data folder into subdatasets
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_fp', type=str, required=True, help='dir containing dataset')
    parser.add_argument('--save_to', type=str, required=True, help='dir to save to')
    parser.add_argument('--N', type=int, required=True, help='numper of partitons')
    args = parser.parse_args()

    maybe_mkdir(args.save_to)
    difficulty = {}

    #Dataset 1 metrics
    print('Computing for {}...'.format(args.dataset_fp))
    tfps = os.listdir(args.dataset_fp)
    for i, fp in enumerate(tfps):
        print('computing {} ({}/{})'.format(fp, i+1, len(tfps)), end='\r')
        traj = torch.load(os.path.join(args.dataset_fp, fp))
        heightmaps = traj['next_observation']['heightmap']
        hd = []
        for i in range(1, heightmaps.shape[0]//10):
            hdiff = (traj['observation']['state'][(i-1)*10, 2] - traj['observation']['state'][i*10, 2]).abs()
            hd.append(hdiff)
        difficulty[fp] = sum(hd) / len(hd)

    diff_arr = torch.stack(tuple(difficulty.values())).numpy()
    plt.hist(diff_arr, bins=100)

    partition = np.ones(len(diff_arr)).astype(int) * (args.N-1)
    partition_fps = []

    for i in range(args.N):
        t_low = np.quantile(diff_arr, i/(args.N))
        t_high = np.quantile(diff_arr, (i+1)/(args.N))
        plt.axvline(t_high, c='r')

        partition[(diff_arr >= t_low) & (diff_arr < t_high)] = i
        partition_fp = '{:.2f}-{:.2f}'.format(t_low, t_high)
        partition_fps.append(partition_fp)
        maybe_mkdir(os.path.join(args.save_to, partition_fp))

    for i, p,fp in enumerate(zip(partition, tfps)):
        print('partitioning {} ({}/{})'.format(fp, i+1, len(tfps)), end='\r')
        pfp = partition_fps[p]
        torch.save(torch.load(os.path.join(args.dataset_fp, fp)), os.path.join(args.save_to, pfp, fp))

    plt.title('Partitioning of Difficulty')
    plt.show()
