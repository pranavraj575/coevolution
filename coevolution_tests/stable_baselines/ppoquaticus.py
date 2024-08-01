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

    env = singleQuaticus()
    model = PPO(policy=MlpPolicy, env=env)
    model.learn(10000)
