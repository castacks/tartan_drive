import torch
import copy

from wheeledsim_rl.util.util import *

"""
List of basic functions (like computing reward-to-go) necessary for some RL algos.
"""
def ensemble_rollout(model, obs, acts, input_normalizer, normalize_actions=False):
    """
    given a starting observation and a tensor of actions, generate states and uncertainty
    Args:
        start_obs:The initial observation from which to roll out states
        acts: a txk tensor, where t denotes timestep and k is the act dim.
    returns:
        states (as dict) and their uncertainties.
    """
    T = acts.shape[0]
    n = model.n_models

    if not isinstance(obs, torch.Tensor):
        obs = dict_to_torch(obs, device=model.device)
    x = dict_repeat(obs, [n, 1])

    norms = input_normalizer.normalize({'observation':x, 'action':acts})
    x = norms['observation']
    nu = norms['action'] if normalize_actions else acts

    pred_buf = [copy.deepcopy(x)]
    unc_buf = []
    for t in range(T):
        with torch.no_grad():
            preds, unc = model.predict_with_uncertainty(pred_buf[-1], nu[t].repeat(n, 1), flatten=True, return_mean=False)
            preds = preds[torch.arange(n), torch.arange(n)]
            unc = preds.std(dim=0).mean(dim=-1)
            preds = model.unflatten_obs(preds)
            
        pred_buf.append(copy.deepcopy(preds))
        unc_buf.append(unc)

    unc_buf = torch.stack(unc_buf)
    states = input_normalizer.denormalize({'next_observation':dict_stack(pred_buf[1:], dim=1)})['next_observation']
    return states, unc_buf

def evaluate_model(model, traj, input_normalizer, n_steps=10, return_traj=False, normalize_actions=True):
    """
    Compute RMSE of the model across a trajectory.
    Args:
        model: The model to evaluate
        data: The data to evaluate on. Expected to be sequential, and a dict containing at least:
            observation, action, next_observation
        input_normalizer: The normalizer used to normalize inputs for the model.
        n_steps: How many steps to forward-simulate the model.
    Returns:
        rmse: A [trajlen-n_steps x n_steps] tensor containing the RMSE. T[i, j] is the RMSE for the i-th trajectory point after forward-simulating j steps.
    """
    rmses = []
    start_obses = {k:v[:-n_steps, :] for k, v in traj['observation'].items()} if isinstance(traj['observation'], dict) else traj['observation'][:-n_steps, :]
    acts = torch.stack([traj['action'][i:-n_steps + i] for i in range(n_steps)], dim=0) #[time x batch x act]

    norms = input_normalizer.normalize({'observation':start_obses, 'action':acts})

    obses = [norms['observation']]
    nacts = norms['action'] if normalize_actions else acts

    for t in range(n_steps):
        with torch.no_grad():
            preds = model.predict(obses[-1], nacts[t])
            if len(preds.shape) > 2: #Take mean of ensemble dim if it exists.
                preds = preds.mean(dim=0)
            preds = model.unflatten_obs(preds)

        obses.append(preds)

    gt = torch.stack([model.flatten_obs(traj['next_observation'])[i:-n_steps + i] for i in range(n_steps)], dim=0) #[time x batch x obs]
    preds = dict_stack(obses[1:], dim=0) #[time x batch x obs]
    preds = input_normalizer.denormalize({'observation':preds})['observation']
    preds = model.flatten_obs(preds)

    error = gt - preds
    rmse = error.pow(2).sum(dim=-1).sqrt() #[time x batch]

    if return_traj:
        return rmse.T, preds, gt, acts
    else:
        return rmse.T

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

def compute_reward_to_go(batch):
	"""
	Computes reward-to-go of a batch of trajectories.
	Expects trajs to be in the following form: {'observation':torch.Tensor[t x obs_dim], 'action':torch.Tensor[t x act_dim], 'terminal':torch.Tensor[t], 'reward':torch.Tensor[t], 'next_observation':torch.Tensor[t x obs_dim]}
	This function will add the reward-to-go at each timestep as 'reward_to_go':torch.Tensor[t] to the batch dict.
	"""
	#Note: There's probably a better way to do all this in parallel.
	#Until then, rtg[i] = 0 if t[i] else rtg[i+1] + r[i+1]

	running_rtg = 0
	reward_to_go = torch.zeros(batch['reward'].shape, device = batch['reward'].device)

	for i in range(reward_to_go.shape[0]-1, -1, -1):
		if batch['terminal'][i]:
			running_rtg = 0
		else:
			running_rtg += batch['discounted_reward'][i + 1]
		reward_to_go[i] = running_rtg

	batch['reward_to_go'] = reward_to_go
	return batch	

def compute_returns(batch):
	"""
	Computes the returns of a batch of trajectories and returns it as a tensor
	"""
	rets = []
	running_ret = 0

	for i in range(batch['reward'].shape[0]):
		running_ret += batch['reward'][i]
		
		if batch['terminal'][i]:
			rets.append(running_ret)
			running_ret = 0

	return torch.tensor(rets, device = batch['reward'].device)

def compute_gae(batch, gamma, lam, vf):
	"""
	Does Generalized advantage estimation (Schulman et al. 2015) for the batch. (Can't do the nice cumsum trick from spinning up because there are multiple trajs in a batch)
	"""
	advantages = torch.zeros(batch['reward'].shape, device = batch['reward'].device)
	v_obs = vf(batch['observation']).detach()
	v_nobs = vf(batch['next_observation']).detach()
	rew = batch['reward']
	tds = gamma * v_nobs + rew - v_obs
	running_adv = 0
	for i in range(advantages.shape[0]-1, -1, -1):
		if batch['terminal'][i]:
			running_adv = 0.
		else:
			running_adv = tds[i] + (gamma * lam) * running_adv
		advantages[i] = running_adv
	return advantages
	

def select_actions_from_q(q, acts):
	"""
	Given a tensor of q values batched as [batch x action] and a list of action_idxs of [batch], select the corresponding action from the tensor
	"""
	return q[torch.arange(q.shape[0]), acts].unsqueeze(1)

#From rlkit: https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
