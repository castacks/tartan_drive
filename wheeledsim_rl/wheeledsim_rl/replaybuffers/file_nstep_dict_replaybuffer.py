import torch
import os
from os import path

from wheeledsim_rl.replaybuffers.dict_replaybuffer import DictReplayBuffer
from wheeledsim_rl.util.os_util import maybe_mkdir
from wheeledsim_rl.util.rl_util import split_trajs
from wheeledsim_rl.util.util import dict_to, dict_cat, dict_stack, dict_map

class FileNStepDictReplayBuffer(DictReplayBuffer):
    """
    Replay buffer that uses files to store more than RAM.
    Note that this will trade off speed for storage capacity.
    TODO: Think about idx sampling that reduces the amount of loads per batch.

    Current implementation is about 25x slower than the cpu/gpu-only buf
    """
    def __init__(self, env, base_fp, files_per_batch=-1, overwrite=False, device='cpu'):
        """
        Args:
            env: The env to get data from
            base_fp: The dir to store/read the data from
            overwrite: If true, overwrite the trajs in base_fp, else put those trajs in the buf.
            files_per_batch: The maximum number of files to load per batch. Lower number = faster. If -1, no limit.
            device: Device to return data on
        """
        super(FileNStepDictReplayBuffer, self).__init__(env, 1, device)
        self.base_fp = base_fp

        self.trajcnt = 0
        self.traj_idx_map = {} #Dict that can be used to query which file to load, given a datapoint idx. Stores the LAST idx for that traj
        self.traj_end_idxs = torch.tensor([]).long()
        self.traj_start_idxs = torch.tensor([]).long()
        self.files_per_batch = files_per_batch

        if overwrite:
            maybe_mkdir(base_fp, force=False)
        else:
            traj_fps = sorted([x for x in os.listdir(self.base_fp) if 'traj' in x])
            bad_fps = []
            for i, tfp in enumerate(traj_fps):
                print('Traj {}/{} ({})'.format(i+1, len(traj_fps), tfp), end='\r')
                try:
                    traj = torch.load(path.join(self.base_fp, tfp), map_location='cpu')
                    self.insert(traj)
                except:
                    bad_fps.append(tfp)
            if len(bad_fps) > 0:
                print('\nCould not load files:')
                for bfp in bad_fps:
                    print(path.join(self.base_fp, bfp))

        self.to(self.device)

    def insert(self, samples):
        """
        Expect samples to be passed in as concatentated contiguous trajs
        """
        assert len(samples['action']) == len(samples['reward']) == len(samples['terminal']), \
        "expected all elements of samples to have same length, got: {} (\'returns\' should be a different length though)".format([(k, len(samples[k])) for k in samples.keys()])
        for k in self.buffer['observation'].keys():
            assert k in samples['observation'].keys(), "Expected observation key {} in traj but didn't find it".format(k)

        for traj in split_trajs(samples):
            traj_fp = "traj_{}.pt".format(self.trajcnt)
            nsamples = traj['action'].shape[0]
            torch.save(dict_to(traj, 'cpu'), path.join(self.base_fp, traj_fp))

            self.traj_start_idxs = torch.cat([self.traj_start_idxs, torch.tensor([self.n])])
            self.traj_end_idxs = torch.cat([self.traj_end_idxs, torch.tensor([self.n + nsamples - 1])])
            self.traj_idx_map[self.n + nsamples - 1] = traj_fp
            self.trajcnt += 1
            self.n += nsamples

    def to(self, device):
        self.device = device
        return self

    def sample(self, nsamples, N):
        """
        Get a batch of samples from the replay buffer.
        Index output as: [batch x time x feats]
        """
        if self.files_per_batch == -1:
            sample_idxs = self.compute_sample_idxs(N)
        else:
            sample_idxs = self.compute_sample_idxs_limit(N, self.files_per_batch)

        idxs = sample_idxs[torch.randint(len(sample_idxs), size=(nsamples, ))]

        outs = [self.sample_idxs((idxs + i) % len(self)) for i in range(N)]
        out = dict_stack(outs, dim=1)

        return out

    def sample_idxs(self, idxs):
        """
        This is more complicated in the file-based buffer. Best way (that minimizes number of loads):
        For this to work with multistep, must enforce that the data output is in the right order (idx order)
        1. Batch the idxs into their trajfiles
        2. Make a dict from traj_fp => query pts
        3. Get the data
        4. Merge the data
        5. Move data to device
        """
        idxs = idxs.to('cpu')
        mask = torch.searchsorted(self.traj_end_idxs, idxs)
        end_idxs = self.traj_end_idxs[mask]
        start_idxs = self.traj_start_idxs[mask]
        fps = [self.traj_idx_map[k.item()] for k in end_idxs]
        queries = {}
        batch_idxs = {} #Keep track of where each elem ends up so we can put the order back
        for fp, start_idx, idx, bidx in zip(fps, start_idxs, idxs, range(len(idxs))):
            if fp in queries.keys():
                queries[fp].append((idx - start_idx).item())
                batch_idxs[fp].append(bidx)
            else:
                queries[fp] = [(idx - start_idx).item()]
                batch_idxs[fp] = [bidx]

        query_keys = sorted(queries.keys())
        res = self.initialize_results(len(idxs))
        for k in query_keys:
            idxs = queries[k]
            bidxs = batch_idxs[k]
            traj_fp = path.join(self.base_fp, k)
            traj = torch.load(traj_fp)

            for k in res.keys():
                if (k == 'observation' or k == 'next_observation'):
                    for kk in res[k].keys():
                        res[k][kk][bidxs] = traj[k][kk][idxs]
                else:
                    res[k][bidxs] = traj[k][idxs]

        return dict_to(res, self.device)

    def compute_sample_idxs_limit(self, N, ntrajs):
        """
        For speed, compute sample idxs for a (small) fixed number of trajs
        """
        terminal_idx_idxs = torch.randint(self.traj_end_idxs.shape[0], size=(ntrajs, ))
        terminal_idxs = self.traj_end_idxs[terminal_idx_idxs].to(self.device)
        start_idxs = self.traj_start_idxs[terminal_idx_idxs].to(self.device)
        all_idxs = torch.cat([torch.arange(si, ti+1).to(self.device) for si, ti in zip(start_idxs, terminal_idxs)])

        non_sample_idxs = torch.tensor([]).long().to(self.device)
        for i in range(N-1):
            non_sample_idxs = torch.cat([non_sample_idxs, terminal_idxs - i])

        #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
        combined = torch.cat((all_idxs, non_sample_idxs))
        uniques, counts = combined.unique(return_counts=True)
        sample_idxs = uniques[counts == 1]

        return sample_idxs.long()

    def compute_sample_idxs(self, N):
        all_idxs = torch.arange(len(self)).to(self.device)
        terminal_idxs = self.traj_end_idxs.to(self.device)
        non_sample_idxs = torch.tensor([]).long().to(self.device)
        for i in range(N-1):
            non_sample_idxs = torch.cat([non_sample_idxs, terminal_idxs - i])

        #https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
        combined = torch.cat((all_idxs, non_sample_idxs))
        uniques, counts = combined.unique(return_counts=True)
        sample_idxs = uniques[counts == 1]

        return sample_idxs.long()

    def initialize_results(self, N):
        res = dict_cat([self.buffer] * N, dim=0)
        res = dict_map(res, lambda x:x.squeeze(-1))
        return res

    def __len__(self):
        return self.n

if __name__ == '__main__':
    from wheeledSim.envs.pybullet_sim import WheeledSimEnv
    env = WheeledSimEnv('../../scripts/data_collection/all_modalities.yaml', render=False)
    fp = '../../../datasets/world_models/all_modalities_shocks_imu'
    import pdb;pdb.set_trace()
    buf = FileNStepDictReplayBuffer(env=env, base_fp=fp, overwrite=False)
    buf = buf.to('cuda')
    batch = buf.sample(16, 3)
    print(batch)
