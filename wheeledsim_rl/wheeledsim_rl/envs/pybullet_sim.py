import pybullet
import numpy as np
import torch
import gym

from wheeledSim.simController import simController
#from wheeledSim.FlatRockyTerrain import FlatRockyTerrain

from wheeledRobots.clifford.cliffordRobot import Clifford

class WheeledSimEnv:
    """
    Wrapper class to make Sean's pybullet env compatible with my RL/opt ctrl code.
    """
    def __init__(self,use_images=False, physicsClientId=0,simulationParamsIn={},senseParamsIn={},
                terrainMapParamsIn={},terrainParamsIn={}, explorationParamsIn={}, existingTerrain=None, cliffordParams={}, T=-1, viz=True, device='cpu'):

        self.client = pybullet.connect(pybullet.GUI) if viz else pybullet.connect(pybullet.DIRECT)
        self.robot = Clifford(params=cliffordParams, physicsClientId=self.client)
        if existingTerrain:
            existingTerrain.generate()
        senseParamsIn['senseType'] = 0 if use_images else -1
        self.env = simController(self.robot, self.client, simulationParamsIn, senseParamsIn, terrainMapParamsIn, terrainParamsIn, explorationParamsIn)
        self.T = T
        self.nsteps = 0
        self.use_images = use_images
        self.device = device

    @property
    def observation_space(self):
        state_space = gym.spaces.Box(low = np.ones(13) * -float('inf'), high = np.ones(13) + float('inf'))
        if not self.use_images:
            return state_space
        else:
            image_space = gym.spaces.Box(low=np.ones(self.env.senseParams['senseResolution']) * -float('inf'), high=np.ones(self.env.senseParams['senseResolution']) * float('inf'))
            return gym.spaces.Dict({'state':state_space, 'image':image_space})

    @property
    def action_space(self):
        return gym.spaces.Box(low = -np.ones(2), high=np.ones(2))

    def reset(self):
        self.env.newTerrain()
        self.nsteps = 0
        self.env.controlLoopStep([0., 0.])
        self.env.resetRobot()
        pose = self.env.robot.getPositionOrientation()
        vel = self.robot.getBaseVelocity_body()
        joints = self.robot.measureJoints()
        if self.env.senseParams['recordJointStates']:
            obs = list(pose[0])+list(pose[1])+vel[:] + joints[:]
        else:
            obs = list(pose[0])+list(pose[1])+vel[:]

        if self.use_images:
            hmap = self.env.sensing(pose, senseType=0)
            return {"state":torch.tensor(obs).float().to(self.device), "image":torch.tensor(hmap).float().to(self.device)}

        else:
             return torch.tensor(obs).float().to(self.device)

    def step(self, action):
        #note that the image is sa[1]
        sa, s, sim_t = self.env.controlLoopStep(action)
        obs = {"state":torch.tensor(s[0]).float().to(self.device), "image":torch.tensor(sa[1]).float().to(self.device)} if self.use_images else torch.tensor(s[0]).float().to(self.device)
        self.nsteps += 1
        timeout = (self.T > 0) and (self.nsteps >= self.T)
        return obs, 0., sim_t or timeout, {}

    def to(self, device):
        self.device = device
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt
#    terrain = FlatRockyTerrain()

    import pdb;pdb.set_trace()
    env = WheeledSimEnv(terrainParamsIn={'N':50}, T=30, use_images=True)
    env.reset()

    for _ in range(5):
        t = False
        cnt = 0
        while not t:
            cnt += 1
            a = env.action_space.sample()
            a = [1.0, 0.0]
            o, r, t, i = env.step(a)
            print('STATE = {}, ACTION = {}, t = {}'.format(o, a, cnt))
            plt.imshow(o['image'], origin='lower');plt.show()
            
        env.reset()
        t = False
