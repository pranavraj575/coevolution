import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy
from stable_baselines3.ppo.ppo import PPO

from unstable_baselines3.dqn.DQN import WorkerDQN
from unstable_baselines3.ppo.PPO import WorkerPPO

from unstable_baselines3.better_multi_alg import multi_agent_algorithm
import os, sys, torch

from src.game_outcome import PettingZooOutcomeFn
from src.utils.dict_keys import (DICT_TRAIN,
                                 DICT_IS_WORKER,
                                 DICT_CLONABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_MUTATION_REPLACABLE,
                                 )
from src.coevolver import PettingZooCaptianCoevolution
from pettingzoo import AECEnv, ParallelEnv

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

    def get_action(self, obs, *args, **kwargs):
        if obs == 3 or np.random.random() < self.p:
            self.choice = np.random.randint(3)
        return self.choice


class SingleZooOutcome(PettingZooOutcomeFn):
    def _get_outcome_from_agents(self, agent_choices, index_choices, updated_train_infos, env):
        a, b = agent_choices
        a = a[0]
        b = b[0]
        a_info, b_info = updated_train_infos
        a_info = a_info[0]
        b_info = b_info[0]

        alg = multi_agent_algorithm(policy=MlpPolicy,
                                    env=env,
                                    DefaultWorkerClass=Worker,
                                    worker_infos={'player_0': a_info,
                                                  'player_1': b_info
                                                  },
                                    workers={'player_0': a,
                                             'player_1': b
                                             },
                                    )
        rec = [0, 0]
        obs = [[], []]
        for i in range(1):
            alg.learn_episode(total_timesteps=5)
            hist = alg.env.unwrapped.history
            for h in range(5):
                game = hist[h*2:h*2 + 2]
                diff = (game[0] - game[1])%3
                if diff == 1:
                    rec[0] += 1
                elif diff == 2:
                    rec[1] += 1

            obs[0].append(torch.rand(1))
            obs[1].append(torch.rand(1))

        if True:
            if any([isinstance(c, easy_pred) or isinstance(c, always_0) for c in (a, b)]):
                print(rec)
        if rec[0] == rec[1]:  # the agents tied
            return [
                (.5, []),
                (.5, []),
            ]
        if rec[0] > rec[1]:
            # agent 0 won
            return [
                (1, []),
                (0, []),
            ]

        if rec[0] < rec[1]:
            # agent 1 won
            return [
                (0, []),
                (1, []),
            ]


DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))


def env_constructor(train_infos):
    env = rps_v2.env()
    env.agents = ['player_0', 'player_1']
    return env


trainer = PettingZooCaptianCoevolution(population_sizes=[3,
                                                         3
                                                         ],
                                       outcome_fn_gen=SingleZooOutcome,
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
                                                                  n_steps=64,
                                                                  batch_size=64,
                                                                  gamma=0.,
                                                                  ),
                                                           {DICT_MUTATION_REPLACABLE: False, })
                                       ],
                                       zoo_dir=os.path.join(DIR, 'data', 'rps_zoo_coevolution_test'),
                                       member_to_population=lambda team_idx, member_idx: {team_idx},
                                       )
save_dir = os.path.join(DIR, 'data', 'save', 'rps_coevolution')
if os.path.exists(save_dir):
    trainer.load(save_dir=save_dir)
for epohc in range(1000):
    # print('starting epoch', trainer.info['epochs'])
    trainer.epoch()
    if not (epohc + 1)%100:
        # print('saving')
        trainer.save(save_dir)
        # print('done saving')

trainer.clear()
