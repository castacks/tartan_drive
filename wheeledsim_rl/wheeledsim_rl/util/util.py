"""
General utilities
"""
import torch
import numpy as np
import gym

class DummyEnv:
    """
    Given a traj, construct a dummy environment with the same obs space (needed for some models.)
    """
    def __init__(self, sample_traj):
        self.act_dim = sample_traj['action'].shape[1:]
        self.action_space = gym.spaces.Box(low=np.ones(self.act_dim)*-float('inf'), high=np.ones(self.act_dim)*float('inf'))

        if isinstance(sample_traj['observation'], torch.Tensor):
            self.obs_dim = sample_traj['observation'].shape[1:]
            self.observation_space = gym.spaces.Box(low=np.ones(self.obs_dim)*-float('inf'), high=np.ones(self.obs_dim)*float('inf'))
        elif isinstance(sample_traj['observation'], dict):
            self.obs_dim = {k:v.shape[1:] for k,v in sample_traj['observation'].items()}
            self.observation_space = gym.spaces.Dict({k:gym.spaces.Box(low=np.ones(v)*-float('inf'), high=np.ones(v)*float('inf')) for k,v in self.obs_dim.items()})

    def reset(self):
        return self.observation_space.sample()

    def step(self, act):
        return self.observation_space.sample(), torch.tensor(-float('inf')), torch.tensor(True), {}


def quantile(x, q):
    """
    Implementing quantile because I'm constrained to torch 1.5
    Sorting isn't the most efficent way to do this but it's not worth the time to implement a smart algo.
    """
    xs = torch.sort(x.flatten())[0]
    idx = int(xs.shape[0] * q)
    return xs[idx]

def dict_repeat(d1, dims):
    if isinstance(d1, dict):
        return {k:dict_repeat(v, dims) for k,v in d1.items()}
    else:
        return d1.repeat(*dims)

def dict_map(d1, fn):
    if isinstance(d1, dict):
        return {k:dict_map(v, fn) for k,v in d1.items()}
    else:
        return fn(d1)

def dict_to_torch(d1, device='cpu'):
    if isinstance(d1, dict):
        return {k:dict_to_torch(v, device) for k,v in d1.items()}
    else:
        return torch.tensor(d1).float().to(device)

def dict_to(d1, device):
    if isinstance(d1, dict):
        return {k:dict_to(v, device) for k,v in d1.items()}
    else:
        return d1.to(device)

def dict_clone(d1, device):
    if isinstance(d1, dict):
        return {k:dict_clone(v, device) for k,v in d1.items()}
    else:
        return d1.clone()

def dict_diff(d1, d2):
    """
    Assumes dicts of dicts or tensors and that dicts have same structure.
    """
    out = {}
    for k in d1.keys():
        if isinstance(d1[k], dict):
            out[k] = dict_diff(d1[k], d2[k])
        else:
            out[k] = d1[k] - d2[k]
    return out 

def dict_stack(d_list, dim=0):
    """
    stacks the elems in a list of dicts.
    Assumes all dicts have the same topology.
    """
    if isinstance(d_list[0], dict):
        return {k:dict_stack([d[k] for d in d_list], dim) for k in d_list[0].keys()}
    else:
        return torch.stack(d_list, dim=dim)

def dict_cat(d_list, dim=0):
    """
    stacks the elems in a list of dicts.
    Assumes all dicts have the same topology.
    """
    if isinstance(d_list[0], dict):
        return {k:dict_cat([d[k] for d in d_list], dim) for k in d_list[0].keys()}
    else:
        return torch.cat(d_list, dim=dim)

def multimatmul(*mats):
    if len(mats) == 1:
        return mats[0]
    elif len(mats) == 2:
        return torch.matmul(mats[0], mats[1])
    else:
        return multimatmul(torch.matmul(mats[0], mats[1]), *mats[2:])

def eul_to_quat(roll, pitch, yaw):
    """
    Convert euler angles to quaternion.
    By convention, put scalar first.
    """
    sr = torch.sin(roll) # 0
    cr = torch.cos(roll) # 1
    sp = torch.sin(pitch) # 0
    cp = torch.cos(pitch) # 1
    sy = torch.sin(yaw)
    cy = torch.cos(yaw)

    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    return torch.stack([qw, qx, qy, qz], dim=-1)

def ssm(vec):
    """
    Get the skew-symmetric matrix equivalent to cross-prodict for a (batch of) 3-element vector(s).
    """
    if len(vec.shape) == 1:
        return ssm(vec.unsuqeeze(0)).squeeze()
    
    out = torch.zeros(vec.shape[0], 3, 3, device=vec.device)
    out[:, 0, 1] = -vec[:, 2]
    out[:, 0, 2] = vec[:, 1]
    out[:, 1, 0] = vec[:, 2]
    out[:, 1, 2] = -vec[:, 0]
    out[:, 2, 0] = -vec[:, 1]
    out[:, 2, 1] = vec[:, 0]

    return out

def quaternion_lmult(q):
    """
    Get the left-multiplcation matrix of a quaternion q. Assume scalar component first.
    """
    if len(q.shape) == 1:
        return quaternion_lmult(q.unsuqeeze(0)).squeeze()

    out = torch.zeros(q.shape[0], 4, 4, device=q.device)
    out[:, 0, 0] = q[:, 0]
    out[:, 0, 1:] = -q[:, 1:]
    out[:, 1:, 0] = q[:, 1:]
    out[:, 1:, 1:] = q[:, 0].unsqueeze(1).unsqueeze(2) * torch.eye(3, device=q.device).unsqueeze(0)
    out[:, 1:, 1:] += ssm(q[:, 1:])

    return out

def quaternion_multiply(q1, q2):
    """
    (Batch) multiply two quaternions.
    """
    if len(q1.shape) == 1:
        return quaternion_multiply(q1.unsqueeze(0), q2).squeeze()

    if len(q2.shape) == 1:
        return quaternion_multply(q1, q2.unsqueeze(0)).squeeze()

    assert q1.shape[0] == q2.shape[0] or q1.shape[0] == 1 or q2.shape[0] == 1, "Improper batching. Got {} quaternions as first arg, {} as second. Expects both to be the same or either to be 1.".format(q1.shape[0], q2.shape[0])

    L = quaternion_lmult(q1)
    return torch.bmm(L, q2.unsqueeze(-1)).squeeze(-1)
