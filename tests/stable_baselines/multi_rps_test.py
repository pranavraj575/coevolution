import numpy as np

from pettingzoo.classic import rps_v2

from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3.dqn.dqn import DQN

from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy
from stable_baselines3.ppo.ppo import PPO

from parallel_algs.dqn.DQN import WorkerDQN
from parallel_algs.ppo.PPO import WorkerPPO

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from parallel_algs.parallel_alg import ParallelAlgorithm
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
worker2=zoo.load_worker(worker_key='0')


if isinstance(worker, OffPolicyAlgorithm):
    print(worker.replay_buffer.size())
else:
    print(worker.rollout_buffer.size())

if isinstance(worker2, OffPolicyAlgorithm):
    print(worker2.replay_buffer.size())
else:
    print(worker2.rollout_buffer.size())
worker2.set_env(worker.env)

print('starting thingy here')
thingy.learn(total_timesteps=20)
quit()


work_save = os.path.join(DIR, 'data', 'sb3_save_test', )
replay_save = os.path.join(DIR, 'data', 'sb3_replay_save_test.pkl')
worker.save(work_save)
if isinstance(worker, OffPolicyAlgorithm):
    worker.save_replay_buffer(replay_save)
    print(worker.replay_buffer.size())
else:
    print(worker.rollout_buffer.size())
worker2 = Worker.load(work_save)
if isinstance(worker2, OffPolicyAlgorithm):
    worker2.load_replay_buffer(replay_save)
    print(worker2.replay_buffer.size())
else:
    print(worker2.rollout_buffer.size())
worker2.set_env(worker.env)
thingy.workers['player_0'] = worker2
print('starting thingy here')
thingy.learn(total_timesteps=20)
