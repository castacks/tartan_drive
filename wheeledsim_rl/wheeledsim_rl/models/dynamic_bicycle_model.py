import torch

from torch import sin, cos, tan, atan2, sqrt

from wheeledsim_rl.util.util import dict_to

class DynamicBicycleModel:
    """
    Implements the following dynamics:
        s = [x, y, psi, xdot, ydot, psidot, beta, steer]
        a = [throttle, steer_rate] in [-1, 1]

        xdotdot = 1/m [F_Lr * cos(psi) + F_Lf * cos(delta+psi) - F_Sf * sin(delta + psi) - F_Sr * sin(psi)]
        ydotdot = 1/m [F_Lr * sin(psi) + F_Lf * sin(delta+psi) + F_Sf * cos(delta + psi) + F_Sr * cos(psi)]
        psidotdot = 1/I_zz [F_Sf * l_v * cos(delta) - F_Sr * l_h + F_Lf * l_v * sin(delta)]

        F_Lr = Kp * (Kv*throttle - v)
        F_Lf = 0
        F_S{f, r} = from Hindiyeh 2013.

    Where Kp, Ks, L are hyperparams of the model.
    """

    def __init__(self, hyperparams = {}, device='cpu'):
        self.device = device
        self.hyperparams = {}
        for k, v in self.default_hyperparams.items():
            self.hyperparams[k] = torch.tensor(hyperparams[k] if k in hyperparams.keys() else self.default_hyperparams[k])

        self.hyperparams = dict_to(self.hyperparams, self.device)

    def dynamics(self, state, action):
        x, y, psi, xdot, ydot, psidot, beta, steer = state.T
        throttle, steer_rate = action.T

        Kp = self.hyperparams['Kp']
        Kv = self.hyperparams['Kv']
        Ks = self.hyperparams['Ks']
        Ksr = self.hyperparams['Ksr']
        L = self.hyperparams['L']
        smax = self.hyperparams['steer_max']

        m = self.hyperparams['m']
        Izz = self.hyperparams['I_zz']
        g = self.hyperparams['g']
        mu = self.hyperparams['mu']
        la = self.hyperparams['l_a']
        lb = self.hyperparams['l_b']
        C = self.hyperparams['C']

        vtarget = Kv * throttle
        delta = Ks * steer
        v = torch.hypot(xdot, ydot)

        steerdot = Ksr * steer_rate
        steerdot[(steer > smax) & (steer_rate > 0.)] *= 0.
        steerdot[(steer < -smax) & (steer_rate < 0.)] *= 0.

        F_Lr = Kp * (vtarget - v)
        F_Lf = torch.zeros_like(F_Lr)

        vxbody = xdot * torch.cos(psi) + ydot * torch.sin(psi)
        vybody = xdot * torch.sin(psi) - ydot * torch.cos(psi)

        alpha_f = torch.atan2(vybody + la*psidot, vxbody) - delta
        alpha_r = torch.atan2(vybody - lb*psidot, vxbody)

        load_f = m*g * (la / (la+lb))
        load_r = m*g * (lb / (la+lb))
        fric_f = load_f * mu
        fric_r = load_r * mu

        eta_f = sqrt(fric_f.pow(2) - F_Lf.pow(2)) / fric_f
        eta_r = sqrt(fric_r.pow(2) - F_Lr.pow(2)) / fric_r

        alpha_sl_f = torch.atan2(3*eta_f*fric_f, C)
        alpha_sl_r = torch.atan2(3*eta_r*fric_r, C)

        F_Sf1 = -eta_f * fric_f * alpha_f.sign()
        F_Sf2 = -C * tan(alpha_f) + (C.pow(2) / (3*eta_f*fric_f)) * tan(alpha_f).abs() * tan(alpha_f).sign() - (C.pow(3) / (27*eta_f.pow(2)*fric_f.pow(2))) * tan(alpha_f).pow(3)

        F_Sr1 = -eta_r * fric_r * alpha_r.sign()
        F_Sr2 = -C * tan(alpha_r) + (C.pow(2) / (3*eta_r*fric_r)) * tan(alpha_r).abs() * tan(alpha_r).sign() - (C.pow(3) / (27*eta_r.pow(2)*fric_r.pow(2))) * tan(alpha_r).pow(3)

        F_Sf1[alpha_f.abs() <= alpha_sl_f] = F_Sf2[alpha_f.abs() <= alpha_sl_f]
        F_Sr1[alpha_r.abs() <= alpha_sl_r] = F_Sr2[alpha_r.abs() <= alpha_sl_r]
        F_Sf = F_Sf1
        F_Sr = F_Sr1

        vxbody_reg = vxbody + 1e-6*vxbody.sign() + 1e-6*(vxbody == 0).float()
        betadot = (F_Sf + F_Sr) / (m*vxbody_reg) - psidot
        xdotdot = (1/m) * (F_Lr*torch.cos(psi) + F_Lf*torch.cos(delta+psi) - F_Sf*torch.sin(delta+psi) - F_Sr*torch.sin(psi))
        ydotdot = (1/m) * (F_Lr*torch.sin(psi) + F_Lf*torch.sin(delta+psi) + F_Sf*torch.cos(delta+psi) + F_Sr*torch.cos(psi))
        psidotdot = (1/Izz) * (F_Sf * la * torch.cos(delta) - F_Lr*lb + F_Lf * la * torch.sin(delta))

        return torch.stack([xdot, ydot, psidot, xdotdot, ydotdot, psidotdot, betadot, steerdot], dim=-1)

    def forward(self, state, action, dt=0.1):
        """
        Get next state from current state, action (via RK4)
        """
#        import pdb;pdb.set_trace()
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
        'R':0.1,
        'g':9.81
    }

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = DynamicBicycleModel()
    seqs = torch.ones(1, 30, 2)
    seqs[:, 15:, 1] *= -1
    state = torch.zeros(seqs.shape[0], 8)
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

