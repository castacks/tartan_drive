import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gym
import time

from mpl_toolkits.mplot3d import Axes3D

from wheeledsim_rl.trainers.base import OffPolicyRLAlgorithm
from wheeledsim_rl.losses.no_loss import NoLoss
from wheeledsim_rl.trainers.world_models_trainer import WorldModelsTrainer
from wheeledsim_rl.util.util import dict_clone, dict_map, dict_to

class ProbWorldModelsTrainer(WorldModelsTrainer):
    """
    World models trainer for prob world models (i.e. NLLLoss on states instead of MSE)
    """
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

        try:
            out = self.model.predict(curr, batch['action'], return_info=True)
        except:
            import pdb;pdb.set_trace()

        preds = out['observation']
        dist = out['encoder_dist']

        self.logger.record_tensor("Model std", preds['state'].scale.detach(), prefix="Debug")

        if 'intervention' in self.model.obs_keys:
            self.logger.record_item("Intervention Target Prob", targets['intervention'].mean(), prefix="Debug")
            self.logger.record_tensor("Intervention Pred", preds['intervention'].sigmoid(), prefix="Debug")
            with torch.no_grad():
                intervention_preds = preds['intervention'].sigmoid() > 0.5
                intervention_labels = targets['intervention'] > 0.5
                accuracy = (intervention_preds == intervention_labels).sum() / np.prod(intervention_labels.shape)
                iaccuracy = (intervention_preds & intervention_labels).sum() / intervention_labels.sum()
            self.logger.record_item("Intervention Accuracy", accuracy.item(), prefix="Performance")
            self.logger.record_item("Intervention Pos Accuracy", iaccuracy.item(), prefix="Performance")

        #NLL
        state_log_prob = preds['state'].log_prob(targets['state']).sum(dim=-1)

        losses = {}
        losses["NLL Loss"] = -state_log_prob.mean()

        for k in self.model.obs_keys:
            means = dist[k].mean
            stds = dist[k].scale
            kl_loss = -0.5 * (1 + stds.pow(2).log() - means.pow(2) - stds.pow(2)).sum(dim=-1).mean() * self.vae_beta

            #Still need to enforce KL loss
            losses["{} KL Loss".format(k)] = kl_loss

            if not isinstance(self.losses[k], NoLoss):
                try:
                    losses["{} Reconstruction Loss".format(k)] = self.losses[k].forward(preds[k], targets[k])
                except:
                    import pdb;pdb.set_trace()
                    print('bad')

        loss = sum(losses.values())

        for k,v in losses.items():
            if k in self.avg_losses.keys():
                self.avg_losses[k].append(v.detach().item())
            else:
                self.avg_losses[k] = [v.detach().item()]

        #Try gradient clipping?
        #Error check

        self.opt.zero_grad()
        loss.backward()

        if any([(x.grad.isnan()|x.grad.isinf()).any() for x in self.model.parameters() if x.grad is not None]):
            print('WARNING: NAN GRAD')
            import pdb;pdb.set_trace()
            return

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        self.opt.step()

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
