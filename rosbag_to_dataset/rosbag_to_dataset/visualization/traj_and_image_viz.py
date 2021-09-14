import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def make_plot(traj, odom_topic='/odom', image_topics=['/multisense/left/image_rect_color', '/multisense/depth'], t=0, fig=None, axs=None):
    """
    Side-by side plot showing the traj and action sequences as well as the image at the current timestep.
    """
    naxs = 2 + len(image_topics)

    if fig is None or axs is None:
        fig, axs = plt.subplots(1, naxs, figsize=(4 * naxs + 1, 4))

    for ax in axs:
        ax.cla()

    axs[0].set_title("Position")
    axs[1].set_title("Action")
    for i, topic in enumerate(image_topics):
        axs[i+2].set_title(topic)

    axs[0].set_xlabel('X(m)')
    axs[0].set_ylabel('Y(m)')
    axs[1].set_xlabel('T')
    
    axs[0].plot(traj['observation'][odom_topic][:, 0], traj['observation'][odom_topic][:, 1], label='Traj')
    axs[0].scatter(traj['observation'][odom_topic][t, 0], traj['observation'][odom_topic][t, 1], c='r', marker='x', label='Current')

    T = np.arange(traj['action'].shape[0])

    for i in range(traj['action'].shape[1]):
        axs[1].plot(T, traj['action'][:, i])
        axs[1].scatter(T[t], traj['action'][t, i])
    axs[1].axvline(t, color='k', linestyle='dotted')

    for i, topic in enumerate(image_topics):
        img = np.moveaxis(traj['observation'][topic][t], 0, 2)
        if img.shape[2] != 3:
            img = img[:, :, -1]

        axs[i+2].imshow(img)

    return fig, axs

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', type=str, required=True, help='Data to visualize')
    args = parser.parse_args()

    dataset = np.load(args.data_fp, allow_pickle=True)
    dataset = {'observation':dataset['observation'].item(), 'action':dataset['action'], 'dt':dataset['dt']}
    
    fig, axs = plt.subplots(1, 4, figsize=(4 * 4 + 1, 4))
    anim = FuncAnimation(fig, func = lambda t:make_plot(dataset, odom_topic='/mocap_node/mushr/Odom', image_topics=[], t=t, fig=fig, axs=axs), frames=np.arange(dataset['action'].shape[0]), interval=dataset['dt']*1000)
    plt.show()
