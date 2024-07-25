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
import os, sys

from src.zoo_cage import ZooCage

Worker = WorkerDQN

if issubclass(Worker, DQN):
    MlpPolicy = DQNPolicy
else:
    MlpPolicy = PPOPolicy

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

env = rps_v2.parallel_env()  # render_mode="human")

observations, infos = env.reset()
for agent in env.agents:
    print(env.observation_space('player_0'))
quit()


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


thingy = ParallelAlgorithm(policy=MlpPolicy,
                           parallel_env=env,
                           DefaultWorkerClass=Worker,
                           # buffer_size=1000,
                           worker_info={'player_1': {'train': False}},
                           workers={'player_1': easy_pred()},
                           # learning_starts=10,
                           gamma=0.,
                           # n_steps=200,
                           # batch_size=100,
                           )

zoo = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_test_rps'))

thingy.learn(total_timesteps=200)

print(thingy.workers)
worker = thingy.workers['player_0']
worker: Worker

zoo.overwrite_worker(worker=worker, worker_key='0')
zoo.overwrite_worker(worker=zoo.load_worker(worker_key='0'), worker_key='1')

zoo.activate_worker(0, '0')
zoo.activate_worker(1, '1')
worker2 = zoo.active_workers[1]

if isinstance(worker, OffPolicyAlgorithm):
    print(worker.replay_buffer.size())
elif isinstance(worker, OnPolicyAlgorithm):
    print(worker.rollout_buffer.size())

if isinstance(worker2, OffPolicyAlgorithm):
    print(worker2.replay_buffer.size())
elif isinstance(worker, OnPolicyAlgorithm):
    print(worker2.rollout_buffer.size())
worker2.set_env(worker.env)
thingy.workers['player_0'] = worker2
print('starting thingy here')
thingy.learn(total_timesteps=220)

if isinstance(worker2, OffPolicyAlgorithm):
    print(worker2.replay_buffer.size())
elif isinstance(worker, OnPolicyAlgorithm):
    print(worker2.rollout_buffer.size())

for _ in range(2):
    # pushes the 220 examples from worker2 into file of worker
    worker = zoo.update_worker_buffer(1, '0')
    if isinstance(worker, OffPolicyAlgorithm):
        print(worker.replay_buffer.size())
    elif isinstance(worker, OnPolicyAlgorithm):
        print(worker.rollout_buffer.size())
