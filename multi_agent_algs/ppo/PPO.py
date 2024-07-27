from stable_baselines3.ppo import PPO
from multi_agent_algs.on_policy import OnPolicy
from stable_baselines3.common.utils import get_schedule_fn


class WorkerPPO(PPO, OnPolicy):
    """
    meant to work inside a parallel DQN
    specifially broke the .learn() and .collect_rollout() methods
    now can iterate in a loop while broadcasting the actions taken to the parallel DQN
    """

    def __init__(self, policy, env, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)

    # redefine this to prevent errors with loading

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()
        # do this conditionally to prevent hitting function recursion limit
        # i.e. reloading models a bunch of time makes get_schedule_fn create a chain of lambda functions
        # equivalent to id(id(.....id(f(x)))), equivalent to f(x)
        if isinstance(self.clip_range, (float, int)):
            self.clip_range = get_schedule_fn(self.clip_range)

        # do the same here
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
                self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
