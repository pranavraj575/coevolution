import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy
from stable_baselines3.ppo.ppo import PPO

from parallel_algs.dqn.DQN import WorkerDQN
from parallel_algs.ppo.PPO import WorkerPPO

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from parallel_algs.parallel_alg import ParallelAlgorithm
import os, sys, torch

from src.zoo_cage import ZooCage
from src.game_outcome import PettingZooOutcomeFn, PlayerInfo
from src.utils.dict_keys import *
from src.coevolver import PettingZooCaptianCoevolution
from src.team_trainer import TeamTrainer

Worker = WorkerPPO

if issubclass(Worker, DQN):
    MlpPolicy = DQNPolicy
else:
    MlpPolicy = PPOPolicy


class always_0:
    def get_action(self, *args, **kwargs):
        return 0


class easy_pred:
    def __init__(self, p=.01):
        self.choice = 0
        self.p = p

    def get_action(self, *args, **kwargs):
        if np.random.random() < self.p:
            self.choice = np.random.randint(3)

        return self.choice


class SingleZooOutcome(PettingZooOutcomeFn):
    def _get_outcome_from_agents(self, agent_choices, index_choices, train_infos, env):
        a, b = agent_choices
        a = a[0]
        b = b[0]
        a_info, b_info = train_infos
        a_info = a_info[0]
        b_info = b_info[0]

        par_alg = ParallelAlgorithm(policy=MlpPolicy,
                                    parallel_env=env,
                                    DefaultWorkerClass=Worker,
                                    worker_info_dict={'player_0': a_info,
                                                      'player_1': b_info
                                                      },
                                    workers={'player_0': a,
                                             'player_1': b
                                             },
                                    )
        rec = [0, 0]
        obs = [[], []]
        for i in range(10):
            par_alg.learn_episode(total_timesteps=1)
            hist = par_alg.env.unwrapped.history
            game = hist[:2]
            diff = (game[0] - game[1])%3
            if diff == 1:
                rec[0] += 1
            elif diff == 2:
                rec[1] += 1
            obs[0].append(par_alg.last_observations['player_0'].item())
            obs[1].append(par_alg.last_observations['player_1'].item())

        obs = [torch.tensor(oo).view((-1, 1)) for oo in obs]
        if True:
            if any([isinstance(c, easy_pred) or isinstance(c, always_0) for c in (a, b)]):
                print(rec)
        if rec[0] == rec[1]:  # the agents tied
            return [
                (.5, [PlayerInfo(obs_preembed=obs[0])]),
                (.5, [PlayerInfo(obs_preembed=obs[1])]),
            ]
        if rec[0] > rec[1]:
            # agent 0 won
            return [
                (1, [PlayerInfo(obs_preembed=obs[0])]),
                (0, [PlayerInfo(obs_preembed=obs[1])]),
            ]

        if rec[0] < rec[1]:
            # agent 1 won
            return [
                (0, [PlayerInfo(obs_preembed=obs[0])]),
                (1, [PlayerInfo(obs_preembed=obs[1])]),
            ]


DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))


def env_constructor():
    env = rps_v2.parallel_env()
    env.agents = ['player_0', 'player_1']
    return env


trainer = PettingZooCaptianCoevolution(population_sizes=[3,
                                                         3
                                                         ],
                                       outcome_fn=SingleZooOutcome(),
                                       env_constructor=env_constructor,
                                       worker_constructors=[
                                           lambda i, env: (easy_pred(p=.01), {DICT_TRAIN: False,
                                                                        DICT_CLONABLE: False,
                                                                        DICT_CLONE_REPLACABLE: False,
                                                                        DICT_MUTATION_REPLACABLE: False,
                                                                        DICT_IS_WORKER: False,
                                                                        }),
                                           lambda i, env: (Worker(policy=MlpPolicy,
                                                                  env=env,
                                                                  n_steps=200,
                                                                  batch_size=100,
                                                                  gamma=0.,
                                                                  ),
                                                           {})
                                       ],
                                       zoo_dir=os.path.join(DIR, 'data', '2rps_zoo_coevolution_test'),
                                       worker_constructors_from_env_input=True,
                                       member_to_population=lambda team_idx, member_idx: {team_idx},
                                       )
for epohc in range(1000):
    trainer.epoch()
trainer.kill_zoo()
