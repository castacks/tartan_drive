import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gym
import time

from mpl_toolkits.mplot3d import Axes3D

from wheeledsim_rl.trainers.base import OffPolicyRLAlgorithm
from wheeledsim_rl.util.util import dict_clone

class ContrastiveWorldModelsTrainer(OffPolicyRLAlgorithm):
    """
    Goal is to maximize the dot product between the sensory latent and the forward-simulated latent for the same traj, but minimize for different trajs
    Implement by dotting each latent to each other latent in the batch. Minimize off-diag elems of the resulting vec.
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
        super(ContrastiveWorldModelsTrainer, self).__init__(env, discount, reward_scale, epochs, steps_per_epoch, initial_steps, replay_buffer, model_itrs, batch_size, model.input_normalizer)
        self.policy = policy
        self.model = model
        self.eval_dataset = eval_dataset
        self.collector = collector
        self.opt = opt
        self.n_steps = n_steps
        self.avg_losses = {}
        self.contrastive_scale = contrastive_scale

        if self.initial_steps > 0:
            self.collect_initial_steps()

    def evaluate(self):
        avg_rmse = torch.zeros_like(self.eval_dataset[0]['observation']['state'][0])
        for traj in self.eval_dataset:
            t = torch.randint(len(traj['action'] - self.n_steps), (1,)).item()

            gt = traj['next_observation']['state'][t:t+self.n_steps]

            curr = {k:v[t].unsqueeze(0) for k,v in traj['observation'].items()}

            with torch.no_grad():
                preds = self.model.predict(curr, traj['action'][t:t+self.n_steps].unsqueeze(0), return_info=False)['state'].squeeze()

            avg_rmse += (preds - gt).pow(2).sum(dim=0).sqrt()

        return avg_rmse / len(self.eval_dataset)
     
    def update(self, batch):
        """
        Instead of predicting sensor outputs, feed later observations through the encoder and match the latents.
        """
        targets = batch['next_observation']
        curr = {k:v[:, 0] for k,v in batch['observation'].items()}

        out = self.model.predict(curr, batch['action'], return_info=True)
        preds = out['observation']
        dist = out['encoder_dist']
        latent_preds = out['latent_observation']

        err = (targets['state'] - preds['state'])

        losses = {}
        losses["Reconstruction Loss"] = err.pow(2).mean()

        #Instead of reconstructing, get latents for sensory input and compare to latent obses.
        latent_sensor_preds = self.model.encode_observation(batch['next_observation'])

        #implement the contrastive loss here.
        similarity = (latent_preds.unsqueeze(0) * latent_sensor_preds.unsqueeze(1)).sum(dim=-1)
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        contrastive_loss = torch.stack([torch.nn.functional.cross_entropy(similarity[:, :, t], labels) for t in range(self.n_steps)])
        losses["Contrastive Loss"] = contrastive_loss.mean()

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
        batch = self.replay_buffer.sample(nsamples = self.qf_batch_size, N=self.n_steps)
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

