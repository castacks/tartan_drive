import torch
import copy

from torch import sin, cos, tan
from numpy import pi

from wheeledsim_rl.util.util import dict_to, dict_stack, dict_map

from pytorch3d.transforms import quaternion_apply, quaternion_multiply, quaternion_invert

class WheelRPMKBM:
    """
    KBM where we use the steer command and the wheel RPM to predict
    i.e. 
        v = K * wheel RPM
        xdot = v * cos(psi)
        ydot = v * sin(psi)        
        psidot = v * tan(Ks * steer) / L
    """

    def __init__(self, hyperparams = {}, device='cpu'):
        self.device = device
        self.hyperparams = {}
        for k, v in self.default_hyperparams.items():
            self.hyperparams[k] = torch.tensor(hyperparams[k] if k in hyperparams.keys() else self.default_hyperparams[k])

        self.hyperparams = dict_to(self.hyperparams, self.device)

    def forward(self, obs, action, dt=0.1):
        x, y, z, qx, qy, qz, qw = obs['state'].moveaxis(-1, 0)
        throttle, steer = action.moveaxis(-1, 0)

        Kv = self.hyperparams['Kv']
        Ks = self.hyperparams['Ks']
        Kt = self.hyperparams['Kt']
        L = self.hyperparams['L']

        v = Kv * obs['wheel_rpm'].mean(dim=-1).mean(dim=-1) + Kt * throttle
        delta = Ks * steer 
        psi = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy+qz*qz))

        #Convert to NED (can ignore z)
        psi = psi + pi/2

        xdot = v * cos(psi)
        ydot = v * sin(psi)
        psidot = v * tan(delta) / L

        xnew = x + dt*xdot
        ynew = y + dt*ydot
        znew = z.clone()

        dpsi = dt*psidot
        dq = torch.stack([dpsi.cos(), torch.zeros_like(dpsi), torch.zeros_like(dpsi), dpsi.sin()], dim=-1) #w, x, y, z
        qold = torch.stack([qw, qx, qy, qz], dim=-1)
        qnew = quaternion_multiply(qold, dq)
        qwnew, qxnew, qynew, qznew = qnew.moveaxis(-1, 0) 

        out = dict_map(obs, lambda x:x.clone())

        out['state'] = torch.stack([xnew, ynew, znew, qxnew, qynew, qznew, qwnew], dim=-1)
        out['wheel_rpm'] = torch.ones_like(obs['wheel_rpm']) * v.unsqueeze(-1).unsqueeze(-1) / Kv

        return out

    def predict(self, obs, action, return_info=True, keys=['state']):
        preds = [obs]
        for t in range(action.shape[1]):
            curr_obs = preds[-1]
            preds.append(self.forward(curr_obs, action[:, t]))

        return dict_stack(preds[1:], dim=1)

    def to(self, device):
        self.device=device
        for k in self.hyperparams.keys():
            self.hyperparams[k] = self.hyperparams[k].to(device)
        return self

    default_hyperparams = {
        'Kv':0.1,
        'Ks':0.5,
        'Kt':0.0,
        'L':2.0,
    }

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    model = WheelRPMKBM()
    seqs = torch.ones(1, 30, 2)
    seqs[:, 15:, 1] *= -1
    state = {'state': torch.zeros(seqs.shape[0], 7), 'wheel_rpm':torch.ones(seqs.shape[0], 10, 4) * 10}
    state['state'][:, -1] = np.sin(1.0)
    state['state'][:, -2] = np.cos(1.0)

    res = model.predict(state, seqs)
    res = res['state']
    for traj, acts in zip(res, seqs):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        axs[0].plot(traj[:, 0], traj[:, 1])
        axs[0].plot(traj[0, 0], traj[0, 1], marker='x')

        axs[1].plot(acts[:, 0], label='throttle')
        axs[1].plot(acts[:, 1], label='steer rate')
        axs[1].set_ylim(-1.1, 1.1)
        axs[1].legend()

        axs[2].plot(traj[:, 0], label='x')
        axs[2].plot(traj[:, 1], label='y')
        axs[2].plot(traj[:, 2], label='psi')
        axs[2].plot(traj[:, 3], label='v')
        axs[2].plot(traj[:, 4], label='steer')
        axs[2].legend()

        plt.show()

