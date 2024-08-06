import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

from unstable_baselines3.ppo.PPO import WorkerPPO
from unstable_baselines3.dqn.DQN import WorkerDQN


from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from unstable_baselines3.common.auto_multi_alg import AutoMultiAgentAlgorithm
import os, sys

from src.zoo_cage import ZooCage

Worker = WorkerPPO

if issubclass(Worker, DQN):
    MlpPolicy = DQNPolicy
else:
    MlpPolicy = PPOPolicy

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

env = rps_v2.parallel_env()  # render_mode="human")

observations, infos = env.reset()


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


thingy = AutoMultiAgentAlgorithm(policy=MlpPolicy,
                               env=env,
                               DefaultWorkerClass=Worker,
                               worker_infos={'player_1': {'train': False}},
                               workers={'player_1': easy_pred()},
                               gamma=0.,

                               # buffer_size=1000,
                               # learning_starts=10,

                               # n_steps=200,
                               # batch_size=100,
                               )

zoo = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_test_rps'))

thingy.learn(total_timesteps=400)

print(thingy.workers)
worker0 = thingy.workers['player_0']
worker0: Worker

zoo.overwrite_worker(worker=worker0, worker_key='0')
zoo.overwrite_worker(worker=zoo.load_worker(worker_key='0')[0], worker_key='1')

worker1, _ = zoo.load_worker(worker_key='1')

if isinstance(worker0, OffPolicyAlgorithm):
    print(worker0.replay_buffer.size())
elif isinstance(worker0, OnPolicyAlgorithm):
    print(worker0.rollout_buffer.size())

if isinstance(worker1, OffPolicyAlgorithm):
    print(worker1.replay_buffer.size())
elif isinstance(worker0, OnPolicyAlgorithm):
    print(worker1.rollout_buffer.size())
worker1.set_env(worker0.env)
thingy.workers['player_0'] = worker1
print('starting thingy here')
thingy.learn(total_timesteps=20)

if isinstance(worker1, OffPolicyAlgorithm):
    print(worker1.replay_buffer.size())
elif isinstance(worker0, OnPolicyAlgorithm):
    print(worker1.rollout_buffer.size())

for _ in range(2):
    # pushes the 220 examples from worker1 into file of worker0
    worker0, _ = zoo.update_worker_buffer(local_worker=worker1, worker_key='0',env=None)
    if isinstance(worker0, OffPolicyAlgorithm):
        print(worker0.replay_buffer.size())
    elif isinstance(worker0, OnPolicyAlgorithm):
        print(worker0.rollout_buffer.size())
