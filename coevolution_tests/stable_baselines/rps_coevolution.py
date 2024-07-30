import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

from unstable_baselines3 import WorkerPPO, WorkerDQN

from unstable_baselines3.common.auto_multi_alg import AutoMultiAgentAlgorithm
import os, sys, torch

from src.game_outcome import PettingZooOutcomeFn
from src.utils.dict_keys import (DICT_TRAIN,
                                 DICT_IS_WORKER,
                                 DICT_CLONABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_MUTATION_REPLACABLE,
                                 )
from src.coevolver import PettingZooCaptianCoevolution

Worker = WorkerPPO
kwargs = {'gamma': 0.,
          }
if issubclass(Worker, DQN):
    MlpPolicy = DQNPolicy
    kwargs['learning_starts'] = 64
    kwargs['batch_size'] = 64
else:
    MlpPolicy = PPOPolicy
    kwargs['n_steps'] = 64
    kwargs['batch_size'] = 64


class always_0:
    def get_action(self, *args, **kwargs):
        return 0


class easy_pred:
    def __init__(self, p=.01):
        self.choice = 0
        self.p = p

    def get_action(self, obs, *args, **kwargs):
        return 0
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

        alg = AutoMultiAgentAlgorithm(policy=MlpPolicy,
                                      env=env,
                                      # DefaultWorkerClass=Worker,
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
            alg.learn(total_timesteps=5)
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

        if False:
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


pop_sizes = [3, 5]

trainer = PettingZooCaptianCoevolution(population_sizes=pop_sizes,
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
                                                                  **kwargs,
                                                                  ),
                                                           {DICT_TRAIN: True,
                                                            DICT_CLONABLE: True,
                                                            DICT_CLONE_REPLACABLE: True,
                                                            DICT_MUTATION_REPLACABLE: False,
                                                            DICT_IS_WORKER: True,
                                                            })
                                       ],
                                       zoo_dir=os.path.join(DIR, 'data', 'rps_zoo_coevolution_test'),
                                       member_to_population=lambda team_idx, member_idx: {team_idx},
                                       processes=1,
                                       )
save_dir = os.path.join(DIR, 'data', 'save', 'rps_coevolution' + str(tuple(pop_sizes)))
# if os.path.exists(save_dir):
#    trainer.load(save_dir=save_dir)
for epohc in range(1000):
    # print('starting epoch', trainer.info['epochs'])
    trainer.epoch()
    if not (epohc + 1)%100:
        # print('saving')
        trainer.save(save_dir)
        # print('done saving')
    print('easy agents elos:', trainer.classic_elos[:pop_sizes[0]])
    print('elos:', trainer.classic_elos[pop_sizes[0]:])

trainer.clear()
