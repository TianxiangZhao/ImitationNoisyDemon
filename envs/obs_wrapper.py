import math
import operator
from functools import reduce

import ipdb
import numpy as np
import gym
from gym import error, spaces, utils
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, Goal
from gym_minigrid.wrappers import FullyObsWrapper

class ImgFlatObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env,):
        super().__init__(env)

        # imgSize = np.prod((self.env.width, self.env.height, 3))
        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + 4,),
            dtype='uint8'
        )

    def observation(self, obs):
        goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) )][0]
        goal_pos = np.array((int(goal_position/self.height), goal_position%self.width))#pos_y, pos_x
        agent_pos = np.array(self.agent_pos)

        image = obs['image']

        obs = np.concatenate((image.flatten(), goal_pos, agent_pos))

        return obs

class IndexObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env,):
        super().__init__(env)

        # imgSize = np.prod((self.env.width, self.env.height, 3))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4,),
            dtype='uint8'
        )

    def observation(self, obs):
        goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) )][0]
        goal_pos = np.array((int(goal_position/self.height), goal_position%self.width)) # pos_y, pos_x
        agent_pos = np.array(self.agent_pos)#pos_x, pos_y
        #ipdb.set_trace()

        obs = np.concatenate((goal_pos, agent_pos))

        return obs

class DirectImgFlatObsWrapper(gym.core.ObservationWrapper):
    """
    extend ImgFlatObsWrapper with current direction as input at the end
    """
    def __init__(self, env,):
        super().__init__(env)

        # imgSize = np.prod((self.env.width, self.env.height, 3))
        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)
        self.Direction_dict = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize + 6,),
            dtype='uint8'
        )

    def observation(self, obs):
        goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) )][0]
        goal_pos = np.array((int(goal_position/self.height), goal_position%self.width))#pos_y, pos_x
        agent_pos = np.array(self.agent_pos)

        agent_dir = np.array(self.Direction_dict[self.agent_dir])

        image = obs['image']

        obs = np.concatenate((image.flatten(), goal_pos, agent_pos, agent_dir))

        return obs
