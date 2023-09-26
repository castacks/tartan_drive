import torch
import argparse
import matplotlib.pyplot as plt

from torch import optim
from tabulate import tabulate

from wheeledsim_rl.models.kinematic_bicycle_model import KinematicBicycleModel

def observation_to_model_state(obs):
    """
    Need to transform the observations to model state.
    1. Vels to global frame
    2. Quaternion to yaw
    """
    if len(obs.shape) == 1:
        return observation_to_model_state(obs.unsqueeze(0)).squeeze()

    x = obs[:, 0]
    y = obs[:, 1]
    qx = obs[:, 3]
    qy = obs[:, 4]
    qz = obs[:, 5]
    qw = obs[:, 6]
    psi = torch.atan2(2 * (qw*qz + qx*qy), 1 - 2*(qz*qz + qy*qy))

    vxb = obs[:, 7]
    vyb = obs[:, 8]
    v = torch.hypot(vxb, vyb)

    return torch.stack([x, y, psi, v], dim=-1)

def set_sysid_variables(model, known_params={}):
    for param in model.hyperparams.keys():
        if param in known_params.keys():
            model.hyperparams[param] = torch.tensor(known_params[param]).float()
        else:
            model.hyperparams[param] = model.hyperparams[param].float()
            model.hyperparams[param].requires_grad = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_fp', type=str, required=True, help='Path to flat-ground data (as buf)')
    parser.add_argument('--sequence_fp', type=str, required=True, help='Path to the action sequences')
    parser.add_argument('--save_as', type=str, required=True, help='Save the params as this.')

    args = parser.parse_args()

    buf = torch.load(args.data_fp)
    seqs = torch.load(args.sequence_fp)

    model = KinematicBicycleModel(hyperparams={'Kv':1.6, 'Ks':-0.5, 'Kp':15.})
    set_sysid_variables(model, {'L':0.4})

    opt = torch.optim.Adam(model.hyperparams.values(), lr=0.01)
    dt = 0.1
    t = seqs.shape[1]
    losses = []

    for ei in range(5000):
        batch = buf.sample_idxs(torch.arange(len(buf)))
#        batch = buf.sample(64)
        states = observation_to_model_state(batch['observation'])
        actions = batch['action']
        next_states = observation_to_model_state(batch['next_observation'])
        pred_next_states = [states.clone()]
        for ti in range(t):
            u = seqs[actions.squeeze(), ti]
            pred_next_states.append(model.forward(pred_next_states[-1], u, dt=dt))

        pred_next_states = torch.stack(pred_next_states, dim=1)

        errs = pred_next_states[:, -1] - next_states
        loss = errs[:, :2].pow(2).mean().sqrt()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ei % 100 == 0:
            i = 10
            pred_next_states = pred_next_states.detach()

            fig, axs = plt.subplots(1, 2, figsize=(9, 4))
            axs[0].plot(pred_next_states[i, :, 0], pred_next_states[i, :, 1], c='r')
            axs[0].arrow(pred_next_states[i, 0, 0], pred_next_states[i, 0, 1], torch.cos(pred_next_states[i, 0, 2])*0.1, torch.sin(pred_next_states[i, 0, 2])*0.1)
            axs[0].scatter(pred_next_states[i, -1, 0], pred_next_states[i, -1, 1], c='r', marker='x')
            axs[0].scatter(next_states[i, 0], next_states[i, 1], marker='x', c='b')

            axs[1].plot(seqs[actions[i, 0], :, 0], label='throttle')
            axs[1].plot(seqs[actions[i, 0], :, 1], label='steer')
            axs[1].set_ylim(-1.1, 1.1)
            axs[1].legend()

            plt.show()

        table = [[k, v.item()] for k,v in model.hyperparams.items()]
        table.insert(0, ['Loss', loss.item()])
        table.insert(0, ['Itr', ei])
        print(tabulate(table, tablefmt='psql'))
        losses.append(loss.detach().item())
    
    torch.save({k:v.detach() for k,v in model.hyperparams.items()}, args.save_as)
    plt.plot(losses)
    plt.show()
