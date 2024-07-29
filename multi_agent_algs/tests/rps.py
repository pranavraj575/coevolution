import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy
from stable_baselines3.ppo.ppo import PPO

from multi_agent_algs.dqn.DQN import WorkerDQN
from multi_agent_algs.ppo.PPO import WorkerPPO

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from multi_agent_algs.better_multi_alg import multi_agent_algorithm
import os, sys

Worker = WorkerPPO

if issubclass(Worker, DQN):
    MlpPolicy = DQNPolicy
else:
    MlpPolicy = PPOPolicy

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

env = rps_v2.parallel_env()  # render_mode="human")


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


thingy = multi_agent_algorithm(policy=MlpPolicy,
                               env=env,
                               DefaultWorkerClass=Worker,
                               worker_infos={'player_1': {'train': False}},
                               workers={'player_1': easy_pred()},
                               gamma=0.,

                               # buffer_size=1000,
                               # learning_starts=10,

                               n_steps=200,
                               batch_size=100,
                               )

thingy.learn(total_timesteps=400)

print(thingy.workers)
worker0 = thingy.workers['player_0']
worker0: Worker

env = rps_v2.parallel_env(render_mode="human")

thingy = multi_agent_algorithm(policy=MlpPolicy,
                               env=env,
                               DefaultWorkerClass=Worker,
                               worker_infos={'player_1': {'train': False}},
                               workers={'player_1': easy_pred()},
                               gamma=0.,

                               # buffer_size=1000,
                               # learning_starts=10,

                               n_steps=200,
                               batch_size=100,
                               )

thingy.learn(total_timesteps=10)