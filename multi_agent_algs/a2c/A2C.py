from stable_baselines3.a2c import A2C
from multi_agent_algs.on_policy import OnPolicy


class WorkerA2C(A2C, OnPolicy):
    """
    meant to work inside a parallel DQN
    specifially broke the .learn() and .collect_rollout() methods
    now can iterate in a loop while broadcasting the actions taken to the parallel DQN
    """

    def __init__(self, policy, env, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
