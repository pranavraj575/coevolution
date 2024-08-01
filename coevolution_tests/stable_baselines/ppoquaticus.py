from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from repos.pyquaticus.pyquaticus import pyquaticus_v0
from repos.pyquaticus.pyquaticus.utils.rewards import sparse
from repos.pyquaticus.pyquaticus.config import config_dict_std


class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs, *args, **kwargs):
        return self.action_space.sample()


class singleQuaticus(gym.Env):
    def __init__(self, render_mode=None, reward_fn=sparse, oppoenent=None, config_dict=config_dict_std):
        """
        Args:
            render_mode: can be 'human'
        """
        super().__init__()
        reward_config = {
            0: reward_fn,
            1: None
        }
        config_dict["max_screen_size"] = (float('inf'), float('inf'))
        self.envquaticus = pyquaticus_v0.PyQuaticusEnv(render_mode=render_mode,
                                                       reward_config=reward_config,
                                                       team_size=1,
                                                       config_dict=config_dict,
                                                       )
        self.envquaticus.reset()
        self.opp_obs = None
        if oppoenent is None:
            oppoenent = RandPolicy(action_space=self.envquaticus.action_space(agent=1))
        self.opponent = oppoenent
        self.action_space = self.envquaticus.action_space(0)
        self.observation_space = self.envquaticus.observation_space(0)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.envquaticus.step(raw_action_dict={
            0: action,
            1: self.opponent.get_action(obs=self.opp_obs)
        })
        self.opp_obs = observations[1]
        return observations[0], rewards[0], terminations[0], truncations[0], (infos[0] if 0 in infos else {})

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        reset_obj = self.envquaticus.reset()
        if type(reset_obj) == tuple:
            observations, infos = reset_obj
        else:
            observations = reset_obj
            infos = {idx: dict() for idx in observations}
        self.opp_obs = observations[1]
        return (observations[0], infos[0])

    def close(self):
        super().close()
        self.envquaticus.close()


if __name__ == '__main__':
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from experiments.pyquaticus_coevolution import config_dict, custom_rew2
    import os, sys
    import dill as pickle
    import numpy as np
    from src.zoo_cage import ZooCage
    from matplotlib import pyplot as plt

    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))
    model_save_file = os.path.join(DIR, 'data', 'single_agent_test', 'stable_baseline', 'model.zip')
    other_model_file = os.path.join(DIR, 'data', 'single_agent_test', 'unstable_baseline')

    zoo = ZooCage(zoo_dir=os.path.join(other_model_file, 'zoo', 'cage_1'), overwrite_zoo=False)
    other_model, _ = zoo.load_animal(key='0')
    f = open(os.path.join(other_model_file, 'info.pkl'), 'rb')
    info: dict = pickle.load(f)
    f.close()
    other_model: PPO
    print(dir(other_model))

    print(info.keys())
    elos = info['captian_elos']
    print('win prob:', np.exp(elos[1])/(np.sum(np.exp(elos.numpy()))))
    epoch_infos = info['epoch_infos']

    if os.path.exists(model_save_file):
        model = PPO.load(model_save_file)
        print(dir(model))

        quit()

    env = singleQuaticus(config_dict=config_dict, reward_fn=custom_rew2, render_mode=None)
    model = PPO(policy=MlpPolicy, env=env, verbose=1)
    model.learn(other_model._total_timesteps)

    model.save(model_save_file)
