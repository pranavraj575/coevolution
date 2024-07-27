import argparse, os, sys
from repos.pyquaticus.pyquaticus import pyquaticus_v0
import repos.pyquaticus.pyquaticus.utils.rewards as rew
from repos.pyquaticus.pyquaticus.config import config_dict_std

from src.coevolver import PettingZooCaptianCoevolution
from src.game_outcome import PettingZooOutcomeFn, PlayerInfo
from src.utils.dict_keys import (DICT_IS_WORKER,
                                 DICT_TRAIN,
                                 DICT_CLONABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_MUTATION_REPLACABLE,
                                 )
from multi_agent_algs.ppo.PPO import WorkerPPO
from stable_baselines3.ppo import MlpPolicy
from multi_agent_algs.better_multi_alg import multi_agent_algorithm


class CTFOutcome(PettingZooOutcomeFn):
    def _get_outcome_from_agents(self, agent_choices, index_choices, train_infos, env):
        agent_choices = agent_choices[0] + agent_choices[1]
        train_infos = train_infos[0] + train_infos[1]

        # env is set up so the first k agents are team blue and the last k agents are team red
        alg = multi_agent_algorithm(env=env,
                                    workers={i: agent_choices[i] for i in range(len(agent_choices))},
                                    worker_infos={i: train_infos[i] for i in range(len(agent_choices))},
                                    )
        alg.learn(total_timesteps=10000,
                  number_of_eps=1,
                  )
        score = (env.unwrapped.blue_team_score, env.unwrapped.red_team_score)
        if score[0] == score[1]:
            return [
                (.5, [PlayerInfo()]),
                (.5, [PlayerInfo()]),
            ]
        if score[0] > score[1]:
            return [
                (1, [PlayerInfo()]),
                (0, [PlayerInfo()]),
            ]
        if score[0] < score[1]:
            return [
                (0, [PlayerInfo()]),
                (1, [PlayerInfo()]),
            ]
class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs, *args, **kwargs):
        return self.action_space.sample()

config_dict = config_dict_std
config_dict["max_screen_size"] = (float('inf'), float('inf'))
config_dict["max_time"] = 420.


reward_config = {0: rew.sparse, 1: rew.sparse, 5: None}  # Example Reward Config
def env_constructor(render_mode=None):
    return pyquaticus_v0.PyQuaticusEnv(render_mode=render_mode,
                                       reward_config=reward_config,
                                       team_size=1,
                                       config_dict=config_dict,
                                       )

if __name__=='__main__':
    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

    data_folder = os.path.join(DIR, 'data', 'pyquaticus_coevolution')

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--render', action='store_true', required=False,
                        help="Enable rendering")

    args = PARSER.parse_args()

    RENDER_MODE = 'human' if args.render else None



