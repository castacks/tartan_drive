import torch

from torch import optim, sin, cos, tan

from wheeledsim_rl.util.util import dict_to

class KBMKinematics:
    """
    Class for the kinematics of a kinematic bicycle model.
    KBM:
        State space = [x, y, theta]
        Control space = [velocity, yaw]
        Dynamics = {
            xdot = v * cos(theta)
            ydot = v * sin(theta)
            thetadot = v * tan(yaw) / L
        }
    Note that this is different than the KBM in the models folder.
    Needs to have both forward and inverse dynamics.
    """
    def __init__(self, hyperparams = {}, device='cpu'):
        self.device = device
        self.hyperparams = {}
        for k, v in self.default_hyperparams.items():
            self.hyperparams[k] = torch.tensor(hyperparams[k] if k in hyperparams.keys() else self.default_hyperparams[k])

        self.hyperparams = dict_to(self.hyperparams, self.device)
        self.state_dim = 3
        self.control_dim = 2

    def dynamics(self, state, control):
        x, y, psi = state.moveaxis(-1, 0)
        v, delta = control.moveaxis(-1, 0)

        L = self.hyperparams['L']

        xdot = v * cos(psi)
        ydot = v * sin(psi)
        psidot = v * tan(delta) / L

        return torch.stack([xdot, ydot, psidot], dim=-1)

    def forward_dynamics(self, state, control, dt):
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(state + (dt/2)*k1, control)
        k3 = self.dynamics(state + (dt/2)*k2, control)
        k4 = self.dynamics(state + dt*k3, control)

        next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state

    def inverse_dynamics(self, state, next_state, dt, max_itrs=100, k_reg=1.0, lr=1e-2, tol=1e-6, verbose=True):
        """
        Due to the nonlinearity of the dynamics, I will implement the inverse dyanamics as a root-finding problem.
        (i.e. optimization + regularizer)
        """
        controls = torch.zeros(*state.shape[:-1], self.control_dim, requires_grad=True).to(self.device)
#        opt = optim.Adam([controls], lr=lr)
        opt = optim.LBFGS([controls])
        for itr in range(max_itrs):
            def closure():
                if torch.is_grad_enabled():
                    opt.zero_grad()
                preds = self.forward_dynamics(state, controls, dt)
                err = (next_state - preds).pow(2).mean() #dont try to fit steer angle.
                reg = controls.pow(2).mean()
                loss = err + k_reg * reg
                if loss.requires_grad:
                    loss.backward()
                return loss

            """
            preds = self.forward_dynamics(state, controls, dt)
            err = (next_state - preds).view(-1, self.state_dim)[:, :-1].pow(2).mean() #dont try to fit steer angle.
            reg = controls.pow(2).mean()
            loss = err + k_reg * reg
            opt.zero_grad()
            loss.backward()
            """

            opt.step(closure)

            with torch.no_grad():
                preds = self.forward_dynamics(state, controls, dt)
                err = (next_state - preds).view(-1, self.state_dim)[:, :-1].pow(2).mean() #dont try to fit steer angle.
                reg = controls.pow(2).mean()

            if err.sqrt() < tol:
                break

            if verbose:
                print('ITR {}: LOSS = {:.6f}, REG = {:.6f}'.format(itr+1, err.detach().cpu().item(), reg.detach().cpu().item()), end='\r')

        return controls.detach()

    def to(self, device):
        self.device=device
        for k in self.hyperparams.keys():
            self.hyperparams[k] = self.hyperparams[k].to(device)
        return self

    default_hyperparams = {
        'L':0.4,
    }

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = KBMKinematics()

    dt = 0.1
#    controls = torch.zeros(50, model.control_dim)
#    controls = torch.tensor([[5., 0.]]).repeat(50, 1)
#    controls[:, 1] += torch.linspace(-0.7, 0.7,  50)

    controls = torch.rand(50, model.control_dim)
    controls[:, 0] *= 5.
    controls[:, 1] -= 0.5

    start_state = torch.zeros(model.state_dim)

    traj = [start_state]

    for u in controls:
        next_state = model.forward_dynamics(traj[-1], u, dt)
        traj.append(next_state)

    traj = torch.stack(traj, dim=0)

    reconstructed_u = model.inverse_dynamics(traj[:-1], traj[1:], dt, k_reg=0.)
    reconstructed_traj = [start_state]
    for u in reconstructed_u:
        next_state = model.forward_dynamics(reconstructed_traj[-1], u, dt)
        reconstructed_traj.append(next_state)

    reconstructed_traj = torch.stack(reconstructed_traj, dim=0)

    print("STATE ERROR = {:.6f}".format((traj - reconstructed_traj).pow(2).mean().sqrt()))
    print("CONTROL ERROR = {:.6f}".format((controls - reconstructed_u).pow(2).mean().sqrt()))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].plot(traj[:, 0], traj[:, 1], c='b')
    axs[0].plot(traj[0, 0], traj[0, 1], marker='x', label='traj', c='b')
    axs[0].plot(reconstructed_traj[:, 0], reconstructed_traj[:, 1], c='r')
    axs[0].plot(reconstructed_traj[0, 0], reconstructed_traj[0, 1], marker='x', label='reconstructed traj', c='r')
    axs[0].legend()

    axs[1].plot(controls[:, 0], label='vel')
    axs[1].plot(controls[:, 1], label='yaw')
    axs[1].plot(reconstructed_u[:, 0], label='reconstructed vel')
    axs[1].plot(reconstructed_u[:, 1], label='reconstructed yaw')
    axs[1].legend()

    axs[2].plot(traj[:, 0], label='x')
    axs[2].plot(traj[:, 1], label='y')
    axs[2].plot(traj[:, 2], label='psi')
    axs[2].plot(reconstructed_traj[:, 0], label='reconstructed_x')
    axs[2].plot(reconstructed_traj[:, 1], label='reconstructed_y')
    axs[2].plot(reconstructed_traj[:, 2], label='reconstructed_psi')
    axs[2].legend()

    plt.show()

