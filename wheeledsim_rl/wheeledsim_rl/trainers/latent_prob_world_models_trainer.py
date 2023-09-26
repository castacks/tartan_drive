import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gym
import time
import copy

from mpl_toolkits.mplot3d import Axes3D

from wheeledsim_rl.trainers.base import OffPolicyRLAlgorithm
from wheeledsim_rl.trainers.latent_world_models_trainer import LatentWorldModelsTrainer
from wheeledsim_rl.util.util import dict_clone, dict_map, dict_to
from wheeledsim_rl.util.rl_util import soft_update_from_to

class LatentProbWorldModelsTrainer(LatentWorldModelsTrainer):
    """
    Prob world models, but use latent-space loss.
    """
    def __init__(
            self,
            env,
            policy,
            model,
            opt,
            replay_buffer,
            collector,
            model_itrs,
            losses = [],
            contrastive_scale=1.0,
            n_steps=10,
            n_eval_steps=10,
            batch_size=32,
            discount=1.0,
            reward_scale=1.0,
            epochs=100,
            steps_per_epoch=50,
            initial_steps=0,
            eval_dataset = None,
            ema=0.0,
            temperature=0.1
            ):
        """
        Args:
            env: The environment whose dynamics we want to model.
            policy: The policy used to collect transitions from the environment
            model: The model we will train to learn the evironment dynamics
            model_trainer: Used to compute losses from training data (trainers are in the trainers folder of this repo)
            model_itrs: How many times to update the model per epoch
            replay_buffer: Replay buffer used to store transitions
            discount: Not used, kept to maintain consistency with superclass
            reward_scale: Same as discount
            epochs: The number of times to collect data from the env
            steps_per_epoch: The number of transitions to collect per epoch
            initial_steps: Seed the buf with this many transitions before training
            eval_dataset: An optional dict of {'observation':, 'action':, 'next_observation':} used to validate model learning on test data.
                Expects tensors to be batched as [batchdim x timedim x featdim]
            w_lr: learning rate on the W matrix
        """
        super(LatentProbWorldModelsTrainer, self).__init__(env, policy, model, opt, replay_buffer, collector, model_itrs, contrastive_scale, n_steps, n_eval_steps, batch_size, discount, reward_scale, epochs, steps_per_epoch, initial_steps, eval_dataset)
        self.ema_tau = ema
        self.temperature = temperature
        self.target_model = copy.deepcopy(self.model)

#        self.latent_rep = torch.nn.Linear(self.model.rnn_hiddensize, self.model.rnn_hiddensize).to(self.model.device)
#        self.latent_rep_opt = torch.optim.Adam(self.latent_rep.parameters())

    def evaluate(self, train_dataset=False):
        avg_rmse = torch.zeros_like(self.eval_dataset[0]['observation']['state'][0], device='cuda')

        if train_dataset:
            batch = self.replay_buffer.sample(nsamples = len(self.eval_dataset), N=self.n_eval_steps+1)
            dataset = [dict_map(batch, lambda x:x[i]) for i in range(len(self.eval_dataset))]
        else:
            dataset = self.eval_dataset

        for traj in dataset:
            traj = dict_to(traj, 'cuda')
            t = torch.randint(len(traj['action']) - self.n_eval_steps, (1,)).item()

            gt = traj['next_observation']['state'][t:t+self.n_eval_steps]

            curr = {k:v[t].unsqueeze(0) for k,v in traj['observation'].items()}

            with torch.no_grad():
                preds = self.model.predict(curr, traj['action'][t:t+self.n_eval_steps].unsqueeze(0), return_info=True)
                preds = preds['observation']['state'].mean.squeeze()

            avg_rmse += (preds - gt).pow(2)[-1].sqrt()

        return avg_rmse / len(self.eval_dataset)
     
    def update(self, batch):
        targets = batch['next_observation']
        curr = {k:v[:, 0] for k,v in batch['observation'].items()}

        out = self.model.predict(curr, batch['action'], return_info=True)
        preds = out['observation']
        dist = out['encoder_dist']
        latent_preds = out['latent_observation']

#        latent_preds = self.latent_rep.forward(latent_preds).relu()

        self.logger.record_tensor("Model std", preds['state'].scale.detach(), prefix="Debug")

        #NLL
        state_log_prob = preds['state'].log_prob(targets['state']).sum(dim=-1)

        losses = {}
        losses["NLL Loss"] = -state_log_prob.mean()

        #Instead of reconstructing, get latents for sensory input and compare to latent obses.
        with torch.no_grad():
#            latent_sensor_preds = self.target_model.encode_observation(batch['next_observation'])
            latent_sensor_preds = self.target_model.predict(dict_map(batch['observation'], lambda x:x.flatten(end_dim=1)) , batch['action'].view(-1, self.model.act_dim).unsqueeze(1), return_info=True)
            latent_sensor_preds = latent_sensor_preds['latent_observation'].view(self.qf_batch_size, self.n_steps, -1)
#            latent_sensor_preds = self.latent_rep.forward(latent_sensor_preds).relu()

        #Implement a contrastive loss (this is NCE)
        similarity = (latent_preds.unsqueeze(0) * latent_sensor_preds.unsqueeze(1)).sum(dim=-1)
        norm = torch.linalg.norm(latent_preds, dim=-1).unsqueeze(0) * torch.linalg.norm(latent_sensor_preds, dim=-1).unsqueeze(1)
        logits = similarity / (norm * self.temperature)

        labels = torch.arange(similarity.shape[0], device=similarity.device)
        contrastive_loss = torch.stack([torch.nn.functional.cross_entropy(logits[:, :, t], labels) for t in range(self.n_steps)])
        latent_reg = latent_preds.pow(2).mean()

        losses["Contrastive Loss"] = contrastive_loss.mean()
        losses['Latent Reg Loss'] = 0. * latent_reg

        loss = sum(losses.values())

        for k,v in losses.items():
            if k in self.avg_losses.keys():
                self.avg_losses[k].append(v.detach().item())
            else:
                self.avg_losses[k] = [v.detach().item()]

        #Try gradient clipping?
        #Error check

        self.opt.zero_grad()
#        self.latent_rep_opt.zero_grad()
        loss.backward()

        if any([(x.grad.isnan()|x.grad.isinf()).any() for x in self.model.parameters() if x.grad is not None]):
            print('WARNING: NAN GRAD')
            import pdb;pdb.set_trace()
            return

        torch.nn.utils.clip_grad_value_(self.model.parameters(), 10.0)
        self.opt.step()
#        self.latent_rep_opt.step()

        soft_update_from_to(self.model, self.target_model, self.ema_tau)

    def train_iteration(self):
        self.current_epoch += 1
        t = time.time()

        self.logger.record_item("Epoch", self.current_epoch)

        if self.steps_per_epoch > 0:
            trajs = self.collector.collect_steps(self.steps_per_epoch, finish_traj = True, deterministic = False)
            self.replay_buffer.insert(trajs)

            self.logger.record_item("Num Episodes", self.logger.get(prefix='', field='Num Episodes', default=0) + trajs['terminal'].float().sum().item())
            self.logger.record_item("Num Added Episodes", trajs['terminal'].float().sum().item())
            self.logger.record_item("Total Steps", self.logger.get(prefix='', field='Total Steps', default=0) + trajs['terminal'].shape[0])

        collect_time = time.time() - t

        nitrs = self.qf_itrs
        for qi in range(nitrs):
            print('itr {}/{}'.format(qi+1, nitrs), end='\r')
            batch = self.collect()

            t = time.time()
            self.update(batch)
            update_time = time.time() - t

            self.logger.record_item("QF Itr", qi)
            self.logger.record_item('Collect time', collect_time, prefix = 'Timing')
            self.logger.record_item('Update time', update_time, prefix = 'Timing')

            t = time.time()
            log_time = time.time() - t
            self.logger.record_item('Log Time', log_time, prefix= 'Timing')

        for k, v in self.avg_losses.items():
            self.logger.record_item(k, torch.tensor(v).mean(), prefix="Performance")

        eval_feature_wise_rmse = self.evaluate(train_dataset=False)
        train_feature_wise_rmse = self.evaluate(train_dataset=True)

        self.logger.record_item("Eval RMSE Features", np.around(eval_feature_wise_rmse.cpu().numpy(), 4), prefix="Performance")
        self.logger.record_item("Eval RMSE", eval_feature_wise_rmse.mean().item(), prefix="Performance")
        self.logger.record_item("Train RMSE", train_feature_wise_rmse.mean().item(), prefix="Performance")

        self.logger.record_item("Return Mean", -eval_feature_wise_rmse.mean().detach().item(), prefix="Performance")

        self.avg_losses = {}

        self.log()

class BilinearLatentProbWorldModelsTrainer(LatentProbWorldModelsTrainer):
    """
    Same as contrastive but use the Info NCE loss. (i.e. qWx.T instead of qx.T)
    """
    def __init__(
            self,
            env,
            policy,
            model,
            opt,
            replay_buffer,
            collector,
            model_itrs,
            contrastive_scale=1.0,
            n_steps=10,
            n_eval_steps=10,
            batch_size=32,
            discount=1.0,
            reward_scale=1.0,
            epochs=100,
            steps_per_epoch=50,
            initial_steps=0,
            eval_dataset = None,
            W_lr=1e-4,
            ema=0.0
            ):
        """
        Args:
            env: The environment whose dynamics we want to model.
            policy: The policy used to collect transitions from the environment
            model: The model we will train to learn the evironment dynamics
            model_trainer: Used to compute losses from training data (trainers are in the trainers folder of this repo)
            model_itrs: How many times to update the model per epoch
            replay_buffer: Replay buffer used to store transitions
            discount: Not used, kept to maintain consistency with superclass
            reward_scale: Same as discount
            epochs: The number of times to collect data from the env
            steps_per_epoch: The number of transitions to collect per epoch
            initial_steps: Seed the buf with this many transitions before training
            eval_dataset: An optional dict of {'observation':, 'action':, 'next_observation':} used to validate model learning on test data.
                Expects tensors to be batched as [batchdim x timedim x featdim]
            w_lr: learning rate on the W matrix
        """
        super(BilinearLatentProbWorldModelsTrainer, self).__init__(env, policy, model, opt, replay_buffer, collector, model_itrs, contrastive_scale, n_steps, n_eval_steps, batch_size, discount, reward_scale, epochs, steps_per_epoch, initial_steps, eval_dataset)
        self.W_lr = W_lr
        self.W = torch.rand(self.model.rnn_hiddensize, self.model.rnn_hiddensize, requires_grad=True, device=self.model.device)
        self.w_opt = torch.optim.Adam([self.W], lr=self.W_lr)
        self.ema_tau = ema
        self.target_model = copy.deepcopy(self.model)

    def update(self, batch):
        targets = batch['next_observation']
        curr = {k:v[:, 0] for k,v in batch['observation'].items()}

        out = self.model.predict(curr, batch['action'], return_info=True)
        preds = out['observation']
        dist = out['encoder_dist']
        latent_preds = out['latent_observation']

        self.logger.record_tensor("Model std", preds['state'].scale.detach(), prefix="Debug")
        self.logger.record_tensor("W", self.W.detach(), prefix="Debug")

        #NLL
        state_log_prob = preds['state'].log_prob(targets['state']).sum(dim=-1)

        losses = {}
        losses["NLL Loss"] = -state_log_prob.mean()

        #Instead of reconstructing, get latents for sensory input and compare to latent obses.
        with torch.no_grad():
#            latent_sensor_preds = self.model.encode_observation(batch['next_observation'])
            latent_sensor_preds = self.target_model.predict(dict_map(batch['observation'], lambda x:x.flatten(end_dim=1)) , batch['action'].view(-1, self.model.act_dim).unsqueeze(1), return_info=True)
            latent_sensor_preds = latent_sensor_preds['latent_observation'].view(self.qf_batch_size, self.n_steps, -1)

        #Implement infoNCE
        similarity = torch.matmul(latent_preds.unsqueeze(-2).unsqueeze(1), torch.matmul(self.W, latent_sensor_preds.unsqueeze(-1).unsqueeze(0))).squeeze() #[batch x batch x time]
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        contrastive_loss = torch.stack([torch.nn.functional.cross_entropy(similarity[:, :, t], labels) for t in range(self.n_steps)])
        latent_reg = latent_preds.pow(2).mean()

        losses["Contrastive Loss"] = contrastive_loss.mean() * self.contrastive_scale
        losses['Latent Reg Loss'] = 0. * latent_reg

        loss = sum(losses.values())

        for k,v in losses.items():
            if k in self.avg_losses.keys():
                self.avg_losses[k].append(v.detach().item())
            else:
                self.avg_losses[k] = [v.detach().item()]

        self.opt.zero_grad()
        self.w_opt.zero_grad()
        loss.backward()

        if any([(x.grad.isnan()|x.grad.isinf()).any() for x in self.model.parameters() if x.grad is not None]):
            print('WARNING: NAN GRAD')
            return


        torch.nn.utils.clip_grad_value_(self.model.parameters(), 10.0)
        self.opt.step()
        self.w_opt.step()

        #Soft target update
        soft_update_from_to(self.model, self.target_model, self.ema_tau)
