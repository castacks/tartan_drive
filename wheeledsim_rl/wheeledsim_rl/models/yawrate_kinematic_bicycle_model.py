import torch

from torch import sin, cos, tan

from wheeledsim_rl.util.util import dict_to

class YawrateKinematicBicycleModel:
    """
    Implements the following dynamics:
        s = [x, y, psi, v, steer]
        a = [throttle, steer_rate] in [-1, 1]

        xdot = v * cos(psi)
        ydot = v * sin(psi)        
        psidot = v * tan(Ks * steer) / L
        vdot = Kp * (Kv * vtarget - v)
        steerdot = steer_rate

    Where Kp, Ks, L are hyperparams of the model.
    """

    def __init__(self, hyperparams = {}, device='cpu'):
        self.device = device
        self.hyperparams = {}
        for k, v in self.default_hyperparams.items():
            self.hyperparams[k] = torch.tensor(hyperparams[k] if k in hyperparams.keys() else self.default_hyperparams[k])

        self.hyperparams = dict_to(self.hyperparams, self.device)

    def dynamics(self, state, action):
        x, y, psi, v, steer = state.T
        throttle, steer_rate = action.T

        Kp = self.hyperparams['Kp']
        Kv = self.hyperparams['Kv']
        Ks = self.hyperparams['Ks']
        Ksr = self.hyperparams['Ksr']
        L = self.hyperparams['L']
        smax = self.hyperparams['steer_max']

        vtarget = Kv * throttle
        delta = Ks * steer

        xdot = v * cos(psi)
        ydot = v * sin(psi)
        psidot = v * tan(delta) / L
        vdot = Kp * (vtarget - v)

        steerdot = Ksr * steer_rate
        steerdot[(steer > smax) & (steer_rate > 0.)] *= 0.
        steerdot[(steer < -smax) & (steer_rate < 0.)] *= 0.

        return torch.stack([xdot, ydot, psidot, vdot, steerdot], dim=-1)

    def forward(self, state, action, dt=0.1):
        """
        Get next state from current state, action (via RK4)
        """
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + (dt/2)*k1, action)
        k3 = self.dynamics(state + (dt/2)*k2, action)
        k4 = self.dynamics(state + dt*k3, action)

        next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state

    def to(self, device):
        self.device=device
        for k in self.hyperparams.keys():
            self.hyperparams[k] = self.hyperparams[k].to(device)
        return self

    default_hyperparams = {
        'Kp':10.0,
        'Ks':-0.5,
        'Kv':2.0,
        'L':0.4,
        'Ksr':5.0, 
        'steer_max':1.0,
        'mu':0.55,
        'C':1200,
        'm':10.,
        'I_zz':0.4,
        'l_a':0.2,
        'l_b':0.2,
        'R':0.1
    }

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = KinematicBicycleModel()
#    seqs = torch.load('../policies/sequences.pt')
#    seqs = (torch.rand(5, 30, 2)-0.5) * 2
    seqs = torch.ones(1, 30, 2)
    seqs[:, 15:, 1] *= -1
    state = torch.zeros(seqs.shape[0], 5)
    res = [state]
    for t in range(seqs.shape[1]):
        cs = res[-1]
        res.append(model.forward(cs, seqs[:, t]))

    import pdb;pdb.set_trace()
    res = torch.stack(res, dim=1)
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

