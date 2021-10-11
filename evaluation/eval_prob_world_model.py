import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from util.rl_util import split_trajs
from util.util import dict_map, dict_to
from util.os_util import str2bool

def viz(gt_traj, pred_traj, gt_latent_traj, pred_latent_traj, ti=4, reconstruction=True):
    """
    Need to plot:
        state error (overhead and over time)
        actions
        reconstruction error over time
        On a separate pane, a 2M x ti grid of gt observations and reconstructions. ti=interp points
    """
    M = len(pred_traj.keys()) - 1
    T = gt_traj['action'].shape[0]
    tidxs = np.linspace(0, T-1, ti).astype(int)

    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 12))
    fig2, axs2 = plt.subplots(ti, 2*M, figsize=(4*M, 2*ti)) if reconstruction else plt.subplots(ti, 1, figsize=(8, 2*ti))

    start = gt_traj['observation']['state'][0, :3]
    gt_acts = gt_traj['action']
    gt_traj = gt_traj['next_observation']

    #Plot traj
    axs1[0, 0].plot(gt_traj['state'][:, 0], gt_traj['state'][:, 1], c='g', marker='.', label='Actual')
    axs1[0, 0].plot(pred_traj['state'].mean[:, 0], pred_traj['state'].mean[:, 1], c='r', marker='.', label='Mean Prediction')
    axs1[0, 0].plot(pred_traj['state'].mean[:, 0] + pred_traj['state'].scale[:, 0], pred_traj['state'].mean[:, 1] + pred_traj['state'].scale[:, 1], c='b', marker='.', linestyle='dotted', label='Prediction + sigma')
    axs1[0, 0].plot(pred_traj['state'].mean[:, 0] - pred_traj['state'].scale[:, 0], pred_traj['state'].mean[:, 1] - pred_traj['state'].scale[:, 1], c='y', marker='.', linestyle='dotted', label='Prediction - sigma')
    axs1[0, 0].scatter(start[0], start[1], c='b', label='start', marker='x')
    axs1[0, 0].set_xlabel('X(m)')
    axs1[0, 0].set_ylabel('Y(m)')
    axs1[0, 0].legend()
    axs1[0, 0].set_title('Traj')

    #Plot 3d traj
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.plot(gt_traj['state'][:, 0], gt_traj['state'][:, 1], gt_traj['state'][:, 2], c='g', marker='.', label='Actual')
    ax3.plot(pred_traj['state'].mean[:, 0], pred_traj['state'].mean[:, 1], pred_traj['state'].mean[:, 2], c='r', marker='.', label='Mean Prediction')
    ax3.scatter(start[0], start[1], start[2], c='b', label='start', marker='x')
    ax3.set_xlabel('X(m)')
    ax3.set_ylabel('Y(m)')
    ax3.set_zlabel('Z(m)')
    ax3.legend()
    ax3.set_title('Traj')

    #Plot traj by time
    colors = 'rgbkmy'
    statelabels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    for i in range(gt_traj['state'].shape[1]):
#        axs1[0, 1].plot(gt_traj['state'][:, i], linestyle='solid', c=colors[i%6])
#        axs1[0, 1].plot(pred_traj['state'].mean[:, i], linestyle='dashed', c=colors[i%6])
#        axs1[0, 1].plot(pred_traj['state'].mean[:, i] + pred_traj['state'].scale[:, i], linestyle='dotted', c=colors[i%6])
#        axs1[0, 1].plot(pred_traj['state'].mean[:, i] - pred_traj['state'].scale[:, i], linestyle='dotted', c=colors[i%6])

        axs1[0, 1].plot(pred_traj['state'].scale[:, i], linestyle='dotted', c=colors[i%6], label=statelabels[i])
        axs1[0, 1].text(gt_traj['state'].shape[0], pred_traj['state'].scale[-1, i], statelabels[i], color='k')

    axs1[0, 1].set_ylabel('Value')
    axs1[0, 1].set_xlabel('T')
    axs1[0, 1].set_title('State std vs time')
    axs1[0, 1].legend()

    #Plot acts
    axs1[1, 0].plot(gt_acts[:, 0], label='Throttle')
    axs1[1, 0].plot(gt_acts[:, 1], label='Steer')
    axs1[1, 0].set_ylim(-1.1, 1.1)
    axs1[1, 0].legend()
    axs1[1, 0].set_title('Actions')

    #TODO: Think about plotting state error here
    #Plot error
    for k in pred_traj.keys():
        if k == 'state':
            gt_obs = gt_traj[k]
            pred_obs = pred_traj[k].mean
            time_rmse = (gt_obs - pred_obs).pow(2).sqrt()
            for i in range(time_rmse.shape[-1]):
                axs1[1, 1].plot(time_rmse[:, i], linestyle='dotted', c=colors[i%6], label=statelabels[i])
                axs1[1, 1].text(gt_traj['state'].shape[0], time_rmse[-1, i], statelabels[i], color='k')

        else:
            gt_obs = gt_traj[k]
            pred_obs = pred_traj[k].mean if isinstance(pred_traj[k], torch.distributions.Normal) else pred_traj[k]
            time_rmse = (gt_obs - pred_obs).pow(2)
            while len(time_rmse.shape) > 1:
                time_rmse = time_rmse.mean(dim=-1).sqrt()
            axs1[1, 1].plot(time_rmse, label="{}".format(k))
    axs1[1, 1].legend()
    axs1[1, 1].set_title('Error (RMSE)')

    if reconstruction:
        plot_reconstructions(fig2, axs2, pred_traj, gt_traj, tidxs, M)
    else:
        plot_latent_states(fig2, axs2, pred_latent_traj, gt_latent_traj, tidxs, M)

    plt.show()

def plot_latent_states(fig, axs, pred_latents, gt_latents, tidxs, M):
    X = torch.arange(pred_latents.shape[1])
    for ii, t in enumerate(tidxs):
        gt_latent = gt_latents[t]
        pred_latent = pred_latents[t]

#        for j in range(gt_latent.shape[-1]):
#            axs[ii].scatter(X, gt_latent[:, j], c='r', marker='.', label='Sensor Encoding')
#        for j in range(gt_latent.shape[-1]):
#            axs[ii].scatter(X, pred_latent[:, j], c='b', marker='.', label='State Encoding')

        axs[ii].plot(X, gt_latent, c='r', marker='.', label='Sensor Encoding')
        axs[ii].plot(X, pred_latent, c='b', marker='.', label='State Encoding')
        axs[ii].set_ylabel('T = {}'.format(t+1))

#    axs[0].legend()
    axs[0].set_title('Latent Encodings')
    axs[-1].set_xlabel('Latent Dim')

    """
    timewise_rmse = (pred_latents - gt_latents).pow(2).sum(dim=-1).sqrt()
    rmsefig, rmseax = plt.subplots()
    rmseax.plot(timewise_rmse)
    rmseax.set_title('Latent RMSE')
    rmseax.set_xlabel('T')
    rmseax.set_ylabel('RMSE')
    """

    return fig, axs

def plot_reconstructions(fig, axs, pred_traj, gt_traj, tidxs, M):
    for i, k in enumerate(pred_traj.keys() - {'state'}):
        for ii, t in enumerate(tidxs):
            if k == 'imu' or k == 'wheel_rpm':
                gt_imu = gt_traj[k][t]
                pred_imu = pred_traj[k][t]

                axs[ii, i].plot(gt_imu)
                axs[ii, i+M].plot(pred_imu)

            else:
                gt_img = gt_traj[k][t]
                pred_img = pred_traj[k][t]

                if gt_img.shape[0] == 1:
                    gt_img = gt_img[0]
                    pred_img = pred_img[0]
                else:
                    gt_img = gt_img[:3].permute(1, 2, 0)
                    pred_img = pred_img[:3].permute(1, 2, 0).clamp(0, 1)

                axs[ii, i].imshow(gt_img)
                axs[ii, i+M].imshow(pred_img)

        axs[0, i].set_title('{} GT'.format(k))
        axs[0, i+M].set_title('{} Pred'.format(k))

    return fig, axs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_fp', type=str, required=True, help='Path to the model to eval')
    parser.add_argument('--eval_data_fp', type=str, required=True, help='Path to the replay buffer containing data to eval')
    parser.add_argument('--T', type=int, required=False, default=10, help='Number of timesteps to predict')
    parser.add_argument('--N', type=int, required=False, default=10, help='Number of samples to eval')
    parser.add_argument('--contrastive', type=str2bool, required=False, default=False, help='For contrastive loss, plot latents instead of reconstructions')
    parser.add_argument('--viz', type=str2bool, required=False, default=False, help='Whether to viz or compute')

    args = parser.parse_args()

    model = torch.load(args.model_fp)

    fps = os.listdir(args.eval_data_fp)
    
    err_buf = []

    for e in range(args.N):
        tfp = np.random.choice(fps)
        traj = torch.load(os.path.join(args.eval_data_fp, tfp), map_location='cpu')

        tidx = np.random.randint(traj['action'].shape[0] - args.T)
        
        gt_traj = dict_map(traj, lambda x:x[tidx:tidx+args.T])
        x0 = {k:v[[0]] for k,v in gt_traj['observation'].items()}
        u = gt_traj['action']
        if args.contrastive:
            with torch.no_grad():
                preds = model.predict(x0, u.unsqueeze(0), return_info=True, keys=['state'])
                pred_traj = preds['observation']
                pred_traj = {k:torch.distributions.Normal(loc=v.loc.squeeze(), scale=v.scale.squeeze()) if isinstance(v, torch.distributions.Normal) else v[0] for k,v in pred_traj.items()}

                pred_latent_traj = preds['latent_observation'][0]
#                pred_latent_traj = preds['latent_prior'].mean[0]
#                pred_latent_traj = preds['latent_prior'].probs[0].topk(k=3)[1]
                gt_latent_traj = model.encode_observation(gt_traj['next_observation']) #Not really gt but naming consistency is nice
#                gt_latent_traj = model.get_latent_posterior(gt_traj['next_observation'], preds['hidden_states'][0]).mean
#                gt_latent_traj = model.get_latent_posterior(gt_traj['next_observation'], preds['hidden_states'][0]).probs.topk(k=3)[1]
        else:
            with torch.no_grad():
                pred_traj = model.predict(x0, u.unsqueeze(0), keys=None)
                pred_traj = {k:torch.distributions.Normal(loc=v.loc.squeeze(), scale=v.scale.squeeze()) if isinstance(v, torch.distributions.Normal) else v[0] for k,v in pred_traj.items()}
                pred_latent_traj = None
                gt_latent_traj = None

        if args.viz:
            viz(gt_traj, pred_traj, gt_latent_traj, pred_latent_traj, reconstruction= (not args.contrastive))

#        position_error = torch.linalg.norm(pred_traj['state'].mean[-1] - gt_traj['next_observation']['state'][-1])
        position_error = (pred_traj['state'].mean - gt_traj['next_observation']['state']).pow(2)[-1].sum()
        err_buf.append(position_error)

    err_buf = torch.stack(err_buf)
    print(err_buf.mean().sqrt())
