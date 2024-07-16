import gymnasium
import numpy as np

def conform_shape(obs, obs_space):
    if obs_space.shape != obs.shape:
        if obs_space.shape[1:] == obs.shape[:2] and obs_space.shape[0] == obs.shape[2]:
            return np.transpose(obs, (2, 0, 1))
    return obs

class DumEnv(gymnasium.Env):
    def __init__(self, action_space, obs_space, ):
        self.action_space = action_space
        self.observation_space = obs_space

    def reset(self, *, seed=None, options=None, ):
        # need to implement this as setting up learning takes in an obs from here for some reason
        return self.observation_space.sample(), {}

    def step(self, action):
        raise NotImplementedError


