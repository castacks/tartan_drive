import os
import torch
import numpy as np
import pandas as pd
import pickle

from wheeledsim_rl.util.os_util import maybe_mkdir
from wheeledsim_rl.util.util import dict_to
from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer

class Experiment:
    """
    Wrapper around RL algorithms that drives the training and handles all the IO stuff. (i.e. making directories, saving networks, recording performance, etc.)    
    """
    def __init__(self, algo, name, experiment_filepath = '', save_every=10, save_logs_every=100, save_best=True, train_data_fp='', buffer_cycle=-1):
        self.algo = algo
        self.name = name
        self.base_fp = os.path.join(os.getcwd(), experiment_filepath, name)
        self.log_fp = os.path.join(self.base_fp, '_log')
        self.save_every = save_every
        self.save_logs_every = save_logs_every
        self.df = None
        self.save_best = save_best
        self.best_return = -np.inf
        self.train_data_fp = train_data_fp
        self.buffer_cycle = buffer_cycle

    def build_experiment_dir(self):
        if os.path.exists(self.base_fp):
            i = input('Directory {} already exists. input \'q\' to stop the experiment (and anything else to keep going).'.format(self.base_fp))
            if i.lower() == 'q':
                exit(0)
        maybe_mkdir(self.base_fp)
        maybe_mkdir(self.log_fp)

    def save_networks(self, dir_fp):
        maybe_mkdir(dir_fp)
        self.write_summary(dir_fp)
        for k, net in self.algo.networks.items():
            d = net.device
            torch.save(net.to('cpu'), os.path.join(dir_fp, '{}.cpt'.format(k)))
            net.to(d)

    def save_env(self):
        #Kinda questionable if this is a good idea, but it'll let me make videos with arbitrary envs by looking at the experiment output.
        env_fp = os.path.join(self.base_fp, 'env.cpt')
        pickle.dump(self.algo.env, open(env_fp, 'wb'))

    def update_log(self):
        if self.df is None:
            self.df = self.algo.logger.dump_dataframe()
        else:
            self.df = pd.concat([self.df, self.algo.logger.dump_dataframe()], ignore_index=True)

    def save_log(self):
        print('Saving log to {}...'.format(self.log_fp))
        if os.path.exists(os.path.join(self.log_fp, "log.csv")):
            self.df.to_csv(os.path.join(self.log_fp, "log.csv"), mode='a', header=False)
        else:
            self.df.to_csv(os.path.join(self.log_fp, "log.csv"))    
        self.df = None

    def write_summary(self, summary_fp):
        with open(os.path.join(summary_fp, '_summary'), 'w') as f:
            f.write('itr = {}\nmean ret = {}'.format(self.algo.current_epoch, self.algo.logger.get(prefix='Performance', field='Return Mean')))

    def run(self):
        self.build_experiment_dir()
#        self.save_env()
        torch.save(self.algo.replay_buffer, os.path.join(self.base_fp, 'buffer.pt')) #Saving the buf for offline MBRL.
        print(len(self.algo.replay_buffer))
        for e in range(self.algo.total_epochs):
            self.algo.train_iteration()
            self.update_log()
            if self.algo.current_epoch % self.save_every == 0:
                self.save_networks(dir_fp = os.path.join(self.base_fp, "itr_{}".format(self.algo.current_epoch)))

            if self.algo.current_epoch % self.save_logs_every == 0:
                self.save_log()

            if self.save_best and self.algo.logger.get(prefix='Performance', field='Return Mean') > self.best_return:
                self.best_return = self.algo.logger.get(prefix='Performance', field='Return Mean')
                self.save_networks(dir_fp = os.path.join(self.base_fp, "_best"))

            self.save_networks(dir_fp = os.path.join(self.base_fp, "_latest"))

            if self.buffer_cycle > 0 and self.train_data_fp != '' and (self.algo.current_epoch % self.buffer_cycle)==0:
                #Change the buffer for larger datasets
                print('cycling buffer...')
                d = self.algo.replay_buffer.device
                n = 0
                fps = os.listdir(self.train_data_fp)
                #Don't totally fill the buf up
                while n < self.algo.replay_buffer.capacity - 100:
                    print('{}/{}'.format(n, self.algo.replay_buffer.capacity), end='\r')
                    traj = torch.load(os.path.join(self.train_data_fp, np.random.choice(fps)))
                    traj = dict_to(traj, d)
                    self.algo.replay_buffer.insert(traj)
                    n += traj['action'].shape[0]
                print('\nloaded {} new steps'.format(n))

