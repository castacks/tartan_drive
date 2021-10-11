import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from tslearn.clustering import TimeSeriesKMeans
from sklearn.manifold import TSNE
from pytorch3d.transforms import quaternion_apply, quaternion_multiply, quaternion_invert

from util.util import dict_map, dict_stack, dict_cat

"""
To answer the question of how much do we need to learn from images.
Cluster action sequences. Then find the closest K to each cluster center and plot the correspondin trajs.
"""

def get_displacement(states):
    """
    Get the displacement/rotation from the start state to the goal state
    NOTE: We represent quaternions as [x, y, z, w], but the rotation library is [w, x, y, z]
    TODO: Finite-diff to get initial speed
    """
    pf = states[:, -1, :3] - states[:, 0, :3]
    qf = torch.cat([states[:, -1, [-1]], states[:, -1, 3:6]], dim=-1)
    qi = torch.cat([states[:, 0, [-1]], states[:, 0, 3:6]], dim=-1)

    pf_rot = quaternion_apply(quaternion_invert(qi), pf)
    qf_rot = quaternion_multiply(quaternion_invert(qi), qf)

    vi = torch.linalg.norm(states[:, 1, :3] - states[:, 0, :3], dim=-1) / 0.1
    
    return {
            'disp':torch.cat([pf_rot, qf_rot], dim=-1),
            'speed':vi
            }

def get_subsequences(tfp, N, T):
    """
    Extract N subsequences of length T from the traj at tfp
    """
    traj = torch.load(tfp, map_location='cpu')
    traj['observation'] = {'state':traj['observation']['state']}
    traj['next_observation'] = {'state':traj['next_observation']['state']}
    max_idx = traj['action'].shape[0] - T
    idxs = torch.randint(max_idx, size=(N, ))
    subseqs = dict_stack([dict_map(traj, lambda x:x[i:i+T]) for i in idxs], dim=0)

    subseqs['observation']['state'] = get_displacement(subseqs['observation']['state'])

    #Rotate by yaw to align start point
    return subseqs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='the directory containing the data to cluster')
    parser.add_argument('--N', type=int, required=False, default=10000, help='number of subsequences to use')
    parser.add_argument('--T', type=int, required=False, default=10, help='length of each subsequence')
    parser.add_argument('--K', type=int, required=False, default=10, help='number of clusters')
    parser.add_argument('--vK', type=int, required=False, default=5, help='number of velocity bins')
    args = parser.parse_args()


    traj_fps = os.listdir(args.data_dir)
    n_trajs = len(traj_fps)
    #Get subsequences
    n_per_seq = int(args.N / n_trajs)
    
    subsequences = []

    for tfp in traj_fps:
        fp = os.path.join(args.data_dir, tfp)
        seqs = get_subsequences(fp, n_per_seq, args.T)
        subsequences.append(seqs)

    subsequences = dict_cat(subsequences, dim=0)

    acts = subsequences['action'].numpy()
    states = subsequences['observation']['state']['disp'].numpy()
    speeds = subsequences['observation']['state']['speed'].numpy()

#    km = TimeSeriesKMeans(n_clusters=args.K, metric="dtw", max_iter=5, max_iter_barycenter=5, random_state=0, n_init=1, verbose=1, n_jobs=12).fit(acts)
    km = TimeSeriesKMeans(n_clusters=args.K, metric="euclidean", max_iter=500, max_iter_barycenter=5, n_init=1, verbose=1, n_jobs=12).fit(acts)
    distances = km.transform(acts)
    closest_cluster = distances.argmin(axis=-1)

    colors = 'rgbkmyc'

    #TODO: Bin based on velocity
    quantiles = np.linspace(0., 1., args.vK + 1)
    qvals = np.array([np.percentile(speeds, 100*q) for q in quantiles])

    #Make hist
    histogram = plt.hist(speeds, bins=100, histtype='step', density=True)
    for q in qvals[1:-1]:
        plt.axvline(q, c='r')
        plt.text(q, 0.9 * histogram[0].max(), '{:.2f}'.format(q))
    plt.title('Distribution of speeds')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Density')
    plt.show()

    for qi in range(args.vK):
        idxs = (speeds > qvals[qi]) & (speeds < qvals[qi+1])
        sub_state = states[idxs]
        tsne = TSNE(verbose=1)
        sub_tsne_state = tsne.fit_transform(sub_state)
        sub_clusters = closest_cluster[idxs]
        sub_acts = acts[idxs]
        for i, cluster in enumerate(km.cluster_centers_):
            color = colors[i % 7]
            idxs = (sub_clusters == i)

            plt.scatter(sub_tsne_state[idxs, 0], sub_tsne_state[idxs, 1], s=1.)

        #For each plot, find three points in the same cluster that are far in TSNE space.
        #Find three points in a different cluster that are close in TSNE space

        data = {
                'embedding':sub_tsne_state,
                'clusters':sub_clusters,
                'acts':sub_acts
                }

        torch.save(data, 'v_{:.2f}-{:.2f}.pt'.format(qvals[qi], qvals[qi+1]))

        plt.title('TSNE for v = [{:.2f}-{:.2f}] m/s'.format(qvals[qi], qvals[qi+1]))
        plt.show()

    #Plot clusters
    for i, cluster in enumerate(km.cluster_centers_):
        color = colors[i % 7]
        plt.plot(cluster[:, 0], c='b', label='Throttle')
        plt.plot(cluster[:, 1], c='r', label='Steer')
        plt.title('Cluster {}'.format(i+1))
        plt.ylim(-1.1, 1.1)
        plt.xlabel('T')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
