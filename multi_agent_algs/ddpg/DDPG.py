from stable_baselines3.ddpg import DDPG
from multi_agent_algs.off_policy import OffPolicy


class WorkerDDPG(DDPG, OffPolicy):
    """
    meant to work inside a parallel DQN
    specifially broke the .learn() and .collect_rollout() methods
    now can iterate in a loop while broadcasting the actions taken to the parallel DQN
    """

    def __init__(self, policy, env, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
