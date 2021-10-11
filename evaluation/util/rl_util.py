import torch
import copy

def split_trajs(batch):
    """
    Given a batch of temporally contiguous data with resets, return the list of rollouts
    """
    tidxs = torch.nonzero(batch['terminal'])[:, 0].long()
    tidxs = torch.cat([torch.tensor([-1]).to(tidxs.device), tidxs])

    trajs = []
    for start, end in zip(tidxs[:-1], tidxs[1:]):
        trajs.append({k: {kk:vv[start+1:end+1] for kk,vv in v.items()}if isinstance(v, dict) else v[start+1:end+1] for k, v in batch.items()})

    if 'map' in batch.keys():
        for traj in trajs:
            traj['map'] = [batch['map'][traj['map_idx'][0]]]

    return trajs
