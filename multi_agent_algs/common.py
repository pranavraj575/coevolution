import gymnasium
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, \
    is_image_space_channels_first

def conform_shape(obs, obs_space):
    if len(obs.shape) == 1:
        obs = obs.reshape((1, -1))
    if obs_space.shape != obs.shape:
        if obs_space.shape[1:] == obs.shape[:2] and obs_space.shape[0] == obs.shape[2]:
            return np.transpose(obs, (2, 0, 1))
    if isinstance(obs_space, spaces.Discrete) and not isinstance(obs, int):
        obs = np.array([obs]).flatten()

    return obs


def conform_act_shape(act, act_space):
    act = act.reshape(act_space.shape)
    if isinstance(act_space, spaces.Discrete) and not isinstance(act, int):
        act = act.reshape(1)[0]
    return act


class DumEnv(gymnasium.Env):
    def __init__(self, action_space, obs_space, ):
        self.action_space = action_space
        self.observation_space = obs_space

    def reset(self, *, seed=None, options=None, ):
        # need to implement this as setting up learning takes in an obs from here for some reason
        return self.observation_space.sample(), {}

    def step(self, action):
        raise NotImplementedError
