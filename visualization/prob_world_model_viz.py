import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from matplotlib.animation import FuncAnimation, writers
from pytorch3d.transforms import quaternion_apply, quaternion_invert, quaternion_multiply

from util import dict_map

def init_plt():
    """
    Get figs ready for viz
    Do a 4-pane of rgbmap + pred, hmap, fcam, error
    """
#    M = 2
#    N = 2
#    fig, axs = plt.subplots(N, M, figsize = (M*4 + 1, N*4 + 1))
    fig, axs = plt.subplots(1, 3, figsize = (13, 5))
    return fig, axs.flatten()

def rotate_traj(states, q):
    #Need to rotate preds to start at 0 yaw.
    qs = torch.cat([states[:, [-1]], states[:, -4:-1]], axis=-1)
    ps = states[:, :3] - states[[0], :3]
    p_rot = quaternion_apply(quaternion_invert(q), ps)
    q_rot = quaternion_multiply(quaternion_invert(q), qs)
    return torch.cat([p_rot, q_rot], axis=-1)

def precompute_results(traj, models, model_names, T):
    """
    Pre-compute results to make some plotting easier
    """
    batch = dict_map(traj, lambda x: x.unsqueeze(0))

    preds = {name:[] for name in model_names}
    errors = {name:[] for name in model_names}
    for t in range(traj['action'].shape[0] - T):
        x0 = dict_map(batch['observation'], lambda x:x[:, t])
        u = batch['action'][:, t:t+T]
        gt = dict_map(batch['next_observation'], lambda x:x[:, t:t+T])['state']
        for name, model in zip(model_names, models):
            with torch.no_grad():
                pred = model.predict(x0, u, return_info=False, keys=['state'])['state'].mean
                error = (gt[:, -1] - pred[:, -1]).pow(2).mean().sqrt()
                errors[name].append(error)

                preds[name].append(pred)

    preds = {k:torch.cat(v) for k,v in preds.items()}
    errors = {k:torch.stack(v) for k,v in errors.items()}
    avg_errors = {k:v.cumsum(0)/(1+torch.arange(len(v))) for k,v in errors.items()}
    return preds, avg_errors

def make_plot(viz_traj, traj, preds, errors, t, T, fig, axs):
    for ax in axs:
        ax.cla()

    #plot preds on map.
    x0 = traj['observation']['state'][t, 0]
    y0 = traj['observation']['state'][t, 1]
    q0 = torch.cat([traj['observation']['state'][t, [-1]], traj['observation']['state'][t, 3:6]])

    axs[0].imshow(traj['observation']['rgbmap'][t].permute(1, 2, 0).fliplr()[:, :, [2, 1, 0]], origin='lower', extent=(-5, 5, 0, 10))

    for name, pred in preds.items():
        pred_rot = rotate_traj(pred[t], q0)
        axs[0].plot(pred_rot[:, 0], pred_rot[:, 1], label=name, linewidth=3.)

    gt_rot = rotate_traj(traj['next_observation']['state'][t:t+T], q0)
    axs[0].plot(gt_rot[:, 0], gt_rot[:, 1], label='GT', linewidth=3.)
    axs[0].legend()
    axs[0].set_xlabel('X(m)')
    axs[0].set_ylabel('Y(m)')
    axs[0].set_title('Prediction')

    #Heightmap
    axs[1].imshow(traj['observation']['heightmap'][t, 0].fliplr(), origin='lower', extent=(-5, 5, 0, 10))
    axs[1].set_title('Heightmap')

    #Front cam
    axs[2].imshow(viz_traj['observation']['image_rgb'][t].permute(1, 2, 0)[:, :, [2, 1, 0]])
    axs[2].set_title('Front Camera')

    """
    #Error over time
    for name, error in errors.items():
        axs[3].plot(error[:t], label=name)

    axs[3].legend()
    axs[3].set_xlabel('T')
    axs[3].set_xlabel('Error (RMSE)')
    axs[3].set_title('Avg Error Over Time')
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_fp', type=str, nargs='+', required=True, help='Path to the model to eval')
    parser.add_argument('--model_name', type=str, nargs='+', required=False, help='Names to call models in viz')
    parser.add_argument('--traj_fp', type=str, required=True, help='The traj to visualize')
    parser.add_argument('--viz_traj_fp', type=str, required=False, default=None, help='optional. If provided, should be the same traj as traj_fp, but with upsampled images. Use this traj to visualise instead')
    parser.add_argument('--T', type=int, required=False, default=10, help='Number of timesteps to predict')

    args = parser.parse_args()

    models = [torch.load(fp) for fp in args.model_fp]
    model_names = ['model_{}'.format(i+1) for i in range(len(models))] if len(args.model_name) == 0 else args.model_name

    traj = torch.load(args.traj_fp)
    viz_traj = traj if args.viz_traj_fp is None else torch.load(args.viz_traj_fp)

    preds, errors = precompute_results(traj, models, model_names, args.T)

    fig, axs = init_plt()
    anim = FuncAnimation(fig, func = lambda t:make_plot(viz_traj, traj, preds, errors, t=t, T=args.T, fig=fig, axs=axs), frames=np.arange(traj['action'].shape[0]-args.T), interval=0.1*1000)
    anim.save('{}_T{}.mp4'.format('_'.join(args.model_name), args.T))
#    plt.show()
