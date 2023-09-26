import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gym
import time

from mpl_toolkits.mplot3d import Axes3D

from wheeledsim_rl.trainers.base import OffPolicyRLAlgorithm
from wheeledsim_rl.trainers.world_models_trainer import WorldModelsTrainer
from wheeledsim_rl.util.util import dict_clone, dict_map

class CurriculumWorldModelsTrainer(WorldModelsTrainer):
    """
    World models trainer, but increase the number of train steps over time.
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
            vae_beta=1e-4,
            reconstruction_scale=1.,
            init_n_steps=10,
            n_eval_steps=10,
            batch_size=32,
            discount=1.0,
            reward_scale=1.0,
            epochs=100,
            steps_per_epoch=50,
            initial_steps=0,
            eval_dataset = None,
            increase_every = 100,
            increase_by = 5,
            max_steps=50
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
        super(CurriculumWorldModelsTrainer, self).__init__(env, policy, model, opt, replay_buffer, collector, model_itrs, vae_beta, reconstruction_scale, init_n_steps, n_eval_steps, batch_size, discount, reward_scale, epochs, steps_per_epoch, initial_steps, eval_dataset)

        self.mean_n_steps = self.n_steps
        self.increase_every = increase_every
        self.increase_by = increase_by
        self.max_steps = max_steps

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
        nstep_buf = []
        for qi in range(nitrs):
            self.n_steps = (self.mean_n_steps + torch.randn([1, ]) * 5).clamp(1, self.max_steps).long()
            nstep_buf.append(self.n_steps.clone())

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
        self.logger.record_item("Mean N steps", self.mean_n_steps, prefix="Training")

        nstep_buf = torch.cat(nstep_buf)
        self.logger.record_tensor("N steps", nstep_buf, prefix="Training")

        self.avg_losses = {}

        if (self.current_epoch % self.increase_every) == 0 and self.n_steps < self.max_steps:
            self.mean_n_steps += self.increase_by

        self.log()
