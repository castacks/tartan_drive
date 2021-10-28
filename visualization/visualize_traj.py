import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from matplotlib.animation import FuncAnimation

"""
Script to visualize trajectories. Automatically infer datatypes with the following heuristic:
state = plot x-y and values
1d: plot as lines
2d: plot as lines per step
3d: image.
"""

def init_plt(traj):
    """
    Get figs ready for viz
    """
    topics = list(traj['observation'].keys())
#    topics = ['image_rgb', 'heightmap', 'rgbmap', 'imu']
    n_panes = len(topics) + 2 #Add an extra for top-down traj, actions
    M = int(n_panes//2) + (n_panes%2)
    N = 2
#    M = n_panes
#    N = 1
    fig, axs = plt.subplots(N, M, figsize = (M*4 + 1, N*4 + 1))
    return fig, axs.flatten(), topics

def make_plot(traj, t, topics, fig, axs):
    for ax in axs:
        ax.cla()
        ax.set_box_aspect(1)

    for ax, topic in zip(axs, topics):
        mode = len(traj['observation'][topic][t].shape)
        if 'map' in topic:
            mode = 4
        if topic == 'state':
            plot_data(traj['observation'][topic][:, 2:], t, mode, fig, ax)
        else:
            plot_data(traj['observation'][topic], t, mode, fig, ax)
        ax.set_title(topic)

    #Plot traj and acts
    axs[-2].set_title('Traj')
#    start = max(0, t-50)
    start = 0
    xs = traj['observation']['state'][start:t+1, 0]
    ys = traj['observation']['state'][start:t+1, 1]
    axs[-2].plot(xs, ys, marker='.', c='r')
    axs[-2].scatter(xs[0], ys[0], marker='^', label='start', c='b')
    axs[-2].scatter(xs[-1], ys[-1], marker='x', label='current', c='b')
    axs[-2].legend()

    if xs.max() - xs.min() < 5:
        axs[-2].set_xlim(xs.mean() - 5, xs.mean() + 5)

    if ys.max() - ys.min() < 5:
        axs[-2].set_ylim(ys.mean() - 5, ys.mean() + 5)

    axs[-1].set_title('Cmds')
    throttle = traj['action'][start:t+1, 0]
    steer = traj['action'][start:t+1, 1]
    axs[-1].plot(throttle, label='throttle', linewidth=3.)
    axs[-1].plot(steer, label='steer', linewidth=3.)
    axs[-1].legend()
    axs[-1].set_ylim(-1.1, 1.1)

#    if t > 30:
#        import pdb;pdb.set_trace()

def plot_data(data, t, mode, fig, ax):
    start = max(0, t-50)
    if mode == 1:
        #Plot state history
        ax.plot(data[start:t+1])
    elif mode == 2:
        if data[t].shape[-1] == 6:
            ax.plot(data[t, :, 3:], linewidth=3)
        else:
            ax.plot(data[t], linewidth=3)
    elif mode == 3:
        ax.imshow(data[t].permute(1, 2, 0).squeeze()[:, :, [2, 1, 0]])
    elif mode == 4:
        if data[t].shape[0] == 3:
            ax.imshow(data[t].permute(1, 2, 0).squeeze()[:, :, [2, 1, 0]].fliplr(), origin='lower')
        elif data[t].shape[0] == 2:
            ax.imshow(data[t].permute(1, 2, 0).squeeze()[:, :, 0].fliplr(), origin='lower')
        else:
            ax.imshow(data[t].permute(1, 2, 0).squeeze().fliplr(), origin='lower')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_fp', type=str, required=True, help='The path to the <traj>.pt data file')
    args = parser.parse_args()

    traj = torch.load(args.traj_fp)

    fig, axs, topics = init_plt(traj)

    anim = FuncAnimation(fig, func = lambda t:make_plot(traj, t=t, topics=topics, fig=fig, axs=axs), frames=np.arange(traj['action'].shape[0]), interval=0.1*1000)
    plt.show()
#    anim.save('video.mp4')
