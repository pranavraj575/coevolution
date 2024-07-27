import argparse, os, sys
from repos.pyquaticus.pyquaticus import pyquaticus_v0
import repos.pyquaticus.pyquaticus.utils.rewards as rew
from repos.pyquaticus.pyquaticus.config import config_dict_std

from src.coevolver import PettingZooCaptianCoevolution
from src.game_outcome import PettingZooOutcomeFn
from src.utils.dict_keys import (DICT_IS_WORKER,
                                 DICT_TRAIN,
                                 DICT_CLONABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_MUTATION_REPLACABLE,
                                 )
from multi_agent_algs.ppo.PPO import WorkerPPO
from stable_baselines3.ppo import MlpPolicy
from multi_agent_algs.better_multi_alg import multi_agent_algorithm


class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs, *args, **kwargs):
        return self.action_space.sample()


DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

data_folder = os.path.join(DIR, 'data', 'pyquaticus_coevolution')

PARSER = argparse.ArgumentParser()

PARSER.add_argument('--render', action='store_true', required=False,
                    help="Enable rendering")

args = PARSER.parse_args()

RENDER_MODE = 'human' if args.render else None
#reward_config = {0: rew.custom_v1, 1: rew.custom_v1, 5: None}  # Example Reward Config
reward_config = {0: rew.sparse, 1: rew.sparse, 5: None}  # Example Reward Config

config_dict = config_dict_std
config_dict["max_screen_size"] = (float('inf'), float('inf'))


# config_dict["max_time"]=1000.

def env_constructor():
    return pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE,
                                       reward_config=reward_config,
                                       team_size=1,
                                       config_dict=config_dict,
                                       )


class CTFOutcome(PettingZooOutcomeFn):
    def _get_outcome_from_agents(self, agent_choices, index_choices, train_infos, env):
        team_a, team_b = agent_choices
        agent_choices = agent_choices[0] + agent_choices[1]
        train_infos = train_infos[0] + train_infos[1]

        # env is set up so the first k agents are team blue and the last k agents are team red
        alg = multi_agent_algorithm(env=env,
                                    workers={i: agent_choices[i] for i in range(len(agent_choices))},
                                    worker_infos={i: train_infos[i] for i in range(len(agent_choices))},
                                    )
        steps = alg.learn(total_timesteps=10000,
                          number_of_eps=1,
                          )
        print(steps)
        print(env)
        print(dir(env.unwrapped))
        score = (env.unwrapped.blue_team_score, env.unwrapped.red_team_score)
        print(score)
        quit()


trainer = PettingZooCaptianCoevolution(population_sizes=[1,
                                                         1
                                                         ],
                                       outcome_fn=CTFOutcome(),
                                       env_constructor=env_constructor,
                                       worker_constructors=[
                                           lambda i, env: (WorkerPPO(policy=MlpPolicy,
                                                                     env=env,
                                                                     ),
                                                           {DICT_TRAIN: True,
                                                            DICT_CLONABLE: False,
                                                            DICT_CLONE_REPLACABLE: False,
                                                            DICT_MUTATION_REPLACABLE: False,
                                                            DICT_IS_WORKER: True,
                                                            }
                                                           ),
                                           lambda i, env: (RandPolicy(env.action_space), {DICT_TRAIN: False,
                                                                                          DICT_CLONABLE: False,
                                                                                          DICT_CLONE_REPLACABLE: False,
                                                                                          DICT_MUTATION_REPLACABLE: False,
                                                                                          DICT_IS_WORKER: False,
                                                                                          }),
                                       ],
                                       zoo_dir=os.path.join(DIR, 'data', 'pyquaticus_coevolution'),

                                       #
                                       member_to_population=lambda team_idx, member_idx: {team_idx},
                                       )
