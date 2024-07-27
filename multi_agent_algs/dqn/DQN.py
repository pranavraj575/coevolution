from stable_baselines3.dqn import DQN
from multi_agent_algs.off_policy import OffPolicy


class WorkerDQN(DQN, OffPolicy):
    """
    meant to work inside a parallel DQN
    specifially broke the .learn() and .collect_rollout() methods
    now can iterate in a loop while broadcasting the actions taken to the parallel DQN
    """

    def __init__(self, policy, env, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
