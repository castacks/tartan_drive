import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gym
import time

from mpl_toolkits.mplot3d import Axes3D

from wheeledsim_rl.algos.base import OffPolicyRLAlgorithm

from wheeledsim_rl.policies.model_based_exploration_policy import OneStepExplorationPolicy

class ResidualDynamicsTrainer(OffPolicyRLAlgorithm):
    """
    Generic algorithm for MBRL. Collect experience from some policy, then sample from that experience to train the model.
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
            n_steps=10,
            batch_size=32,
            discount=1.0,
            reward_scale=1.0,
            epochs=100,
            steps_per_epoch=50,
            initial_steps=0,
            eval_dataset = None
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
        """
        super(ResidualDynamicsTrainer, self).__init__(env, discount, reward_scale, epochs, steps_per_epoch, initial_steps, replay_buffer, model_itrs, batch_size, model.input_normalizer)
        self.policy = policy
        self.model = model
        self.eval_dataset = eval_dataset
        self.collector = collector
        self.opt = opt
        self.n_steps = n_steps
        self.avg_losses = {}

        if self.initial_steps > 0:
            self.collect_initial_steps()

    def evaluate(self):
        avg_rmse = torch.zeros_like(self.eval_dataset[0]['observation'][0])
        for traj in self.eval_dataset:
            t = torch.randint(len(traj['observation'] - self.n_steps), (1,)).item()

            gt = traj['next_observation'][t:t+self.n_steps]

            cstate = traj['observation'][[t]]
            preds = []
            for act in traj['action'][t:t+self.n_steps]:
                with torch.no_grad():
                    cstate = self.model.predict(cstate, act.unsqueeze(0))

                preds.append(cstate.clone())

            preds = torch.cat(preds, dim=0)
            avg_rmse += (preds - gt).pow(2).sum(dim=0).sqrt()

        return avg_rmse / len(self.eval_dataset)
     
    def update(self, batch):
        targets = batch['next_observation']
        curr = batch['observation'][:, 0]
        preds = [curr.clone()]

        for t in range(self.n_steps):
            curr = self.model.predict(preds[-1], batch['action'][:, t])
            preds.append(curr)

        preds = torch.stack(preds[1:], dim=-2)
#        err = (targets - preds)[:, -1]
        err = (targets - preds)

        norm_err = err / self.model.output_normalizer.std['observation'].unsqueeze(0)

        losses = {}
#        losses["Reconstruction Loss"] = norm_err.pow(2).mean()
        losses["Reconstruction Loss"] = err.pow(2).mean()

        loss = sum(losses.values())

        self.avg_losses['Unnormalized Reconstruction Loss'] = err.pow(2).mean()
        for k,v in losses.items():
            if k in self.avg_losses.keys():
                self.avg_losses[k].append(v.detach().item())
            else:
                self.avg_losses[k] = [v.detach().item()]

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def collect_initial_steps(self):
#        import pdb;pdb.set_trace()
        trajs = self.collector.collect_steps(n=self.initial_steps, policy=self.policy, deterministic = False)
        self.replay_buffer.insert(trajs)
        self.observation_normalizer.update(trajs)
        targets = self.model.get_training_targets(trajs, flatten=False)
        self.output_normalizer.update({'observation':targets, 'next_observation':targets})
        self.logger.record_item("Total Steps", self.initial_steps)

    def collect(self):
        #TODO: This is a hack until I implement a traj validity check.
        batch = self.replay_buffer.sample(nsamples = self.qf_batch_size, N=self.n_steps+5)
        batch = {k:v[:, :-5] for k,v in batch.items()}
        return batch

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

        feature_wise_rmse = self.evaluate()

        self.logger.record_item("Eval RMSE Features", np.around(feature_wise_rmse.cpu().numpy(), 4), prefix="Performance")
        self.logger.record_item("Eval RMSE", feature_wise_rmse.mean().item(), prefix="Performance")
        self.logger.record_item("Train RMSE", torch.tensor(self.avg_losses['Unnormalized Reconstruction Loss']).sqrt().mean(), prefix="Performance")

        self.logger.record_item("Return Mean", -feature_wise_rmse.mean().detach().item(), prefix="Performance")

        self.avg_losses = {}

        self.log()

    @property
    def networks(self):
        return {
            'model':self.model,
            'input_normalizer':self.model.input_normalizer,
            'output_normalizer':self.model.output_normalizer,
        }

    @property
    def hyperparameters(self):
        pass

    @property
    def default_hyperparameters(self):
        pass

class EnsembleResidualDynamicsTrainer(ResidualDynamicsTrainer):
    def update(self, batch):
        targets = batch['next_observation']
        curr = batch['observation'][:, 0]
        preds = [curr.clone()]

        ensemble_idxs = torch.randint(self.model.n_models, size=(curr.shape[0], ))
        uncertainty = []

        for t in range(self.n_steps):
            curr = self.model.predict(preds[-1], batch['action'][:, t])
            uncertainty.append(curr.std(dim=0).sum(dim=-1).mean().detach())
            curr = curr[ensemble_idxs, torch.arange(curr.shape[1])]
            preds.append(curr)

        preds = torch.stack(preds[1:], dim=-2)
        uncertainty = torch.stack(uncertainty).mean().cpu().item()

        err = (targets - preds)

        losses = {}
        losses["Reconstruction Loss"] = err.pow(2).mean()

        loss = sum(losses.values())

        self.avg_losses['Unnormalized Reconstruction Loss'] = err.pow(2).mean()
        for k,v in losses.items():
            if k in self.avg_losses.keys():
                self.avg_losses[k].append(v.detach().item())
            else:
                self.avg_losses[k] = [v.detach().item()]

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.logger.record_item("Uncertainty", uncertainty, prefix="Uncertainty")

    def evaluate(self):
        avg_rmse = torch.zeros_like(self.eval_dataset[0]['observation'][0])
        avg_unc = 0.
        for traj in self.eval_dataset:
            t = torch.randint(len(traj['observation']) - self.n_steps, (1,)).item()

            gt = traj['next_observation'][t:t+self.n_steps]
            if gt.shape[0] < self.n_steps:
                import pdb;pdb.set_trace()

            cstate = traj['observation'][[t]].repeat(self.model.n_models, 1)
            preds = []
            for act in traj['action'][t:t+self.n_steps]:
                with torch.no_grad():
                    cstate = self.model.predict(cstate, act.repeat(self.model.n_models, 1))
                    cstate = cstate[torch.arange(self.model.n_models), torch.arange(self.model.n_models)]

                preds.append(cstate.clone())

            preds = torch.stack(preds, dim=1)
            avg_unc += preds.std(dim=0).sum(dim=-1)
            avg_rmse += (preds.mean(dim=0) - gt).pow(2).sum(dim=0).sqrt()

        return (avg_rmse / len(self.eval_dataset)), (avg_unc / len(self.eval_dataset))

    def train_iteration(self):
        self.current_epoch += 1
        t = time.time()

        self.logger.record_item("Epoch", self.current_epoch)

        if self.steps_per_epoch > 0:
            trajs = self.collector.collect_steps(self.steps_per_epoch, finish_traj = True, deterministic = False)
            #Garbage line
            self.policy.t = self.policy.T

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

        feature_wise_rmse, time_wise_unc = self.evaluate()

        self.logger.record_item("Eval RMSE Features", np.around(feature_wise_rmse.cpu().numpy(), 4), prefix="Performance")
        self.logger.record_item("Eval RMSE", feature_wise_rmse.mean().item(), prefix="Performance")
        self.logger.record_item("Eval Unc Time", np.around(time_wise_unc.cpu().numpy(), 4), prefix="Performance")
        self.logger.record_item("Eval Unc", time_wise_unc.mean().item(), prefix="Performance")
        self.logger.record_item("Train RMSE", torch.tensor(self.avg_losses['Unnormalized Reconstruction Loss']).sqrt().mean(), prefix="Performance")

        self.logger.record_item("Return Mean", -feature_wise_rmse.mean().detach().item(), prefix="Performance")

        self.avg_losses = {}

        self.log()
