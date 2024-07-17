import numpy as np

from pettingzoo.classic import rps_v2
from pettingzoo import ParallelEnv
from pettingzoo.butterfly import pistonball_v6

from stable_baselines3.dqn.policies import CnnPolicy, MlpPolicy
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.off_policy_algorithm import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from parallel_algs.dqn.DQN import WorkerDQN
from parallel_algs.off_policy import ParallelOffPolicyAlg


env = rps_v2.parallel_env()#render_mode="human")
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


thingy = ParallelOffPolicyAlg(policy=MlpPolicy,
                              parallel_env=env,
                              buffer_size=1000,
                              worker_info={'player_1': {'train': False}},
                              workers={'player_1': easy_pred()},
                              learning_starts=10,
                              gamma=0.,
                              )

thingy.learn(total_timesteps=10000)

quit()

parallel_env = pistonball_v6.parallel_env(render_mode="human", continuous=False, n_pistons=6)
observations, infos = parallel_env.reset(seed=42)

thingy = ParallelDQN(policy=CnnPolicy, parallel_env=parallel_env, buffer_size=1000)
thingy.learn(total_timesteps=10000)

quit()

guy = 'piston_0'
# test = DumEnv(action_space=parallel_env.action_space(guy), obs_space=parallel_env.observation_space(guy))
# DQN(CnnPolicy, env=test, buffer_size=100)
print(observations[guy].shape)

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
    # print([parallel_env.action_space(agent) for agent in parallel_env.agents])
    print(actions)

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    print(rewards[guy])
    print(terminations[guy])
    print(truncations[guy])

parallel_env.close()
