import torch
from numpy import pi

"""
Collection of useful geometry functions foe things like manipulating robot pose, camera, etc.
"""

def eul_to_vecs(x):
    """
    Takes a collection of Euler angles in a tensor (expected last dim is [roll, pitch, yaw])
    and converts to a 6D representation more amenable for learning (no disconinuities)
    The representation [x1, ..., x6] are the x and y axes of the corresponding rotm.

    Args:
        x: An [n x 3] tensor
    Returns: An [n x 6 tensor]
    """
    sa = torch.sin(x[:, 2])
    sb = torch.sin(x[:, 1])
    sg = torch.sin(x[:, 0])
    ca = torch.cos(x[:, 2])
    cb = torch.cos(x[:, 1])
    cg = torch.cos(x[:, 0])

    x1 = ca*cb
    x2 = sa*cb
    x3 = -sb
    y1 = ca*sb*sg - sa*cg
    y2 = sa*sb*sg + ca*cg
    y3 = cb*sg
    z1 = ca*sb*cg + sa*sg
    z2 = sa*sb*cg - ca*sg
    z3 = cb*cg

    out = torch.stack([x1, x2, x3, y1, y2, y3], dim=1)
    return out

def vecs_to_eul(x):
    """
    Inverse of 6D_to_eul. Takes cat'ed x and y vecs of a rotation matrix and returns the resulting euler angles.
    NOTE THAT THERE ARE MULTIPLE SETS OF EULER ANGLES FOR A ROTM. We'll assume pitch within +- pi/2 (a.k.a. vehicle not upside down)
    Args:
        x: An [n x 6] tensor
    Returns: An [n x 3] tensor of Euler angles
    """
    vx = x[:, :3]
    vy = x[:, 3:6]
    vx /= torch.norm(vx, dim=1, keepdim=True)
    vy /= torch.norm(vy, dim=1, keepdim=True)
    vz = torch.cross(vx, vy)
 
    yaw = torch.atan2(x[:, 1], vx[:, 0])
    pitch = torch.atan2(-vx[:, 2], (vy[:, 2].pow(2) + vz[:, 2].pow(2)).sqrt())
    roll = torch.atan2(vy[:, 2], vz[:, 2])

    out = torch.stack([roll, pitch, yaw], dim=1)
    return out

if __name__ == '__main__':
    tfp = '../../../datasets/airsim/traj_dataset_10hz_small/traj_99.pt'
    traj = torch.load(tfp)
    rots = traj['observation']['orientation']
    vecs = eul_to_vecs(rots)
    rots2 = vecs_to_eul(vecs)

    print(rots, end='\n\n')
    print(rots2, end='\n\n')

    print('Transformations working?', torch.allclose(rots, rots2))

    rots = torch.stack([rots[0] + torch.tensor([0., 0., i/100 * 2 * pi]) for i in range(1009)], dim=0)
    vecs = eul_to_vecs(rots)
    rots2 = vecs_to_eul(vecs)

    print(rots[-5:] % (2*pi), end='\n\n')
    print(rots2[-5:], end='\n\n')
    print(rots[-5:] - rots2[-5:])
    print(vecs)

    
