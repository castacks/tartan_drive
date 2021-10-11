import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == '__main__':
    """
    Use the terrain mapping metrics from Fankhauser et al 2018 to quantify how hards a dataset is relative to another
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset1_fp', type=str, required=True, help='dir containing dataset 1')
    parser.add_argument('--dataset2_fp', type=str, required=True, help='dir containing dataset 2')
    args = parser.parse_args()

    smooth1 = []
    slope1 = []
    curv1 = []
    hdiff1 = []
    act_std1 = []

    smooth2 = []
    slope2 = []
    curv2 = []
    hdiff2 = []
    act_std2 = []

    #Dataset 1 metrics
    print('Computing for {}...'.format(args.dataset1_fp))
    tfps = os.listdir(args.dataset1_fp)
    for i, fp in enumerate(tfps):
        print('computing {} ({}/{})'.format(fp, i+1, len(tfps)))
        traj = torch.load(os.path.join(args.dataset1_fp, fp))
        heightmaps = traj['next_observation']['heightmap']
        hd = []
        for i in range(1, heightmaps.shape[0]//10):
            hdiff = (traj['observation']['state'][(i-1)*10, 2] - traj['observation']['state'][i*10, 2]).abs()
            hd.append(hdiff)
        hdiff1.append(sum(hd) / len(hd))
        act_std1.append(sum(traj['action'].std(dim=0)))

    #Dataset 2 metrics
    print('Computing for {}...'.format(args.dataset2_fp))
    tfps = os.listdir(args.dataset2_fp)
    for i, fp in enumerate(tfps):
        print('computing {} ({}/{})'.format(fp, i+1, len(tfps)))
        traj = torch.load(os.path.join(args.dataset2_fp, fp))
        heightmaps = traj['next_observation']['heightmap']
        hd = []
        for i in range(1, heightmaps.shape[0]//10):
            hdiff = (traj['observation']['state'][(i-1)*10, 2] - traj['observation']['state'][i*10, 2]).abs()
            hd.append(hdiff)
        hdiff2.append(sum(hd) / len(hd))
        act_std2.append(sum(traj['action'].std(dim=0)))

    hdiff1 = torch.stack(hdiff1).numpy()
    hdiff2 = torch.stack(hdiff2).numpy()
    act_std1 = torch.stack(act_std1).numpy()
    act_std2 = torch.stack(act_std2).numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].boxplot([hdiff1, hdiff2])
    axs[0].set_title('Height Diff')

    axs[1].boxplot([act_std1, act_std2])
    axs[1].set_title('Act Std')
    
    axs[2].boxplot([curv1, curv2])
    axs[2].set_title('Curv')

    for ax in axs:
        ax.legend()

    plt.show()

    print('Median hdiff of dataset 1:', np.median(hdiff1))
    print('Median hdiff of dataset 2:', np.median(hdiff2))
    print('Frac of trajs in dataset 2 harder than median of dataset 1:', sum(hdiff2 > np.median(hdiff1)) / len(hdiff2))
