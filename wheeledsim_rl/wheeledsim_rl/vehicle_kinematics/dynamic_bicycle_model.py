import torch

from torch import optim, sin, cos, tan

from wheeledsim_rl.util.util import dict_to

class DBMKinematics:
    """
    Use dynamic bicycle model dynamics, but make it so the tire forces are inputs as well. i.e:
    State = [x, y, theta, xdot, ydot, thetadot, steer_angle]
    Forces = [F_lf, F_sf, F_lr, F_sr]
        l/s = longitudinal/lateral
        f/r = front/back

    Dynamics:
        xdotdot = (1/m) * (F_lr*cos(theta) + F_lf*cos(theta+steer_angle) - F_sr*sin(theta) - F_sf*sin(theta+steer_angle))
        ydotdot = (1/m) * (F_lr*sin(theta) + F_lf*sin(theta*steer_angle) + F_sr*cos(theta) - F_sf*cos(theta_steer_angle))
        thetadotdot = (1/Izz) * (F_sf*la*cos(steer_angle)  + F_lf*la*sin(steer_angle) - F_lr*lb)
    """
    def __init__(self, hyperparams = {}, device='cpu'):
        self.device = device
        self.hyperparams = {}
        for k, v in self.default_hyperparams.items():
            self.hyperparams[k] = torch.tensor(hyperparams[k] if k in hyperparams.keys() else self.default_hyperparams[k])

        self.hyperparams = dict_to(self.hyperparams, self.device)
        self.state_dim=7
        self.control_dim=4

    def dynamics(self, state, forces):
        """
        Note that for numerical scaling, the forces in forces are not the actual forces. Instead, we normalize by mass.
        Thus, we have:
            xdotdot = (~F_lr*cos(theta) + ~F_lf*cos(theta+steer_angle) - ~F_sr*sin(theta) - ~F_sf*sin(theta+steer_angle))
            ydotdot = (~F_lr*sin(theta) + ~F_lf*sin(theta*steer_angle) + ~F_sr*cos(theta) - ~F_sf*cos(theta_steer_angle))
            thetadotdot = (~F_sf*la*cos(steer_angle)  + ~F_lf*la*sin(steer_angle) - ~F_lr*lb)
        """
        x, y, psi, xdot, ydot, psidot, steer_angle = state.moveaxis(-1, 0)
        F_lf, F_sf, F_lr, F_sr = forces.moveaxis(-1, 0)

        la = self.hyperparams['l_a']
        lb = self.hyperparams['l_b']

        xdotdot = F_lr*cos(psi) + F_lf*cos(psi+steer_angle) - F_sr*sin(psi) - F_sf*sin(psi+steer_angle)
        ydotdot = F_lr*sin(psi) + F_lf*sin(psi+steer_angle) + F_sr*cos(psi) + F_sf*cos(psi+steer_angle)
        psidotdot = F_sf*la*cos(steer_angle) + F_lf*la*sin(steer_angle) - F_lr*lb
        steer_angle_dot = torch.zeros_like(steer_angle)

        return torch.stack([xdot, ydot, psidot, xdotdot, ydotdot, psidotdot, steer_angle_dot], dim=-1)

    def forward_dynamics(self, state, forces, dt):
        k1 = self.dynamics(state, forces)
        k2 = self.dynamics(state + (dt/2)*k1, forces)
        k3 = self.dynamics(state + (dt/2)*k2, forces)
        k4 = self.dynamics(state + dt*k3, forces)

        next_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state

    def inverse_dynamics(self, state, next_state, dt, max_itrs=100, k_reg=1.0, lr=1e-1, tol=1e-4, verbose=True):
        """
        Get the tire forces that move between two states. 
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
                err = (next_state - preds).view(-1, self.state_dim)[:, :-1].pow(2).mean() #dont try to fit steer angle.
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
        'm':10.,
        'I_zz':0.4,
        'l_a':0.2,
        'l_b':0.2,
    }

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from wheeledsim_rl.vehicle_kinematics.kinematic_bicycle_model import KBMKinematics

    ref_model = KBMKinematics()
#    ref_model = DBMKinematics()
    model = DBMKinematics()

    dt = 0.1
#    controls = torch.zeros(50, ref_model.control_dim)
#    controls = torch.tensor([[0., 0.]]).repeat(50, 1)
#    controls[:, 0] += torch.linspace(0.0, 3.0,  50)
#    controls[:, 1] += torch.linspace(-0.3, 0.3,  50)

    controls = torch.rand(50, ref_model.control_dim)
#    controls[:, 0] *= 5.
#    controls[:, 1] -= 0.5

    start_state = torch.zeros(ref_model.state_dim)

    traj = [start_state]

    for u in controls:
        next_state = ref_model.forward_dynamics(traj[-1], u, dt)
        traj.append(next_state)

    traj = torch.stack(traj, dim=0)

    #Extrapolate rates, steer angle into DBM state.
    xdot = (traj[1:, 0] - traj[:-1, 0]) / dt
    xdot = torch.cat([torch.zeros_like(xdot[[0]]), xdot])
    ydot = (traj[1:, 1] - traj[:-1, 1]) / dt
    ydot = torch.cat([torch.zeros_like(ydot[[0]]), ydot])
    psidot = (traj[1:, 2] - traj[:-1, 2]) / dt
    psidot = torch.cat([torch.zeros_like(psidot[[0]]), psidot])
    steer_angle = torch.cat([controls[:, 1], controls[-1, [1]]])
    traj = torch.cat([traj, torch.stack([xdot, ydot, psidot, steer_angle], dim=-1)], dim=-1)

    reconstructed_u = model.inverse_dynamics(traj[:-1], traj[1:], dt, k_reg=0.)
    reconstruct_start_state = torch.zeros(model.state_dim)
    reconstructed_traj = [reconstruct_start_state]
    for u in reconstructed_u:
        next_state = model.forward_dynamics(reconstructed_traj[-1], u, dt)
        reconstructed_traj.append(next_state)

    reconstructed_traj = torch.stack(reconstructed_traj, dim=0)

    print("STATE ERROR = {:.6f}".format((traj - reconstructed_traj).pow(2).mean().sqrt()))
#    print("CONTROL ERROR = {:.6f}".format((controls - reconstructed_u).pow(2).mean().sqrt()))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].plot(traj[:, 0], traj[:, 1], c='b')
    axs[0].plot(traj[0, 0], traj[0, 1], marker='x', label='traj', c='b')
    axs[0].plot(reconstructed_traj[:, 0], reconstructed_traj[:, 1], c='r')
    axs[0].plot(reconstructed_traj[0, 0], reconstructed_traj[0, 1], marker='x', label='reconstructed traj', c='r')
    axs[0].legend()

    axs[1].plot(controls[:, 0], label='vel')
    axs[1].plot(controls[:, 1], label='yaw')
    axs[1].plot(reconstructed_u[:, 0], label='reconstructed F_lf')
    axs[1].plot(reconstructed_u[:, 1], label='reconstructed F_sf')
    axs[1].plot(reconstructed_u[:, 2], label='reconstructed F_lr')
    axs[1].plot(reconstructed_u[:, 3], label='reconstructed F_sr')
    axs[1].legend()

    axs[2].plot(traj[:, 0], label='x')
    axs[2].plot(traj[:, 1], label='y')
    axs[2].plot(traj[:, 2], label='psi')
    axs[2].plot(reconstructed_traj[:, 0], label='reconstructed_x')
    axs[2].plot(reconstructed_traj[:, 1], label='reconstructed_y')
    axs[2].plot(reconstructed_traj[:, 2], label='reconstructed_psi')
    axs[2].legend()

    plt.show()
