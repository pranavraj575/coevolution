from repos.pyquaticus.pyquaticus.envs.pyquaticus import PyQuaticusEnv
from repos.pyquaticus.pyquaticus.structs import Team
from repos.pyquaticus.pyquaticus.base_policies.base import BaseAgentPolicy


class MyQuaticusEnv(PyQuaticusEnv):
    """
    keep track of observations of each agent
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_record = []

    def step(self, *args, **kwargs):
        obs, rewards, terminated, truncated, info = super().step(*args, **kwargs)
        self.obs_record.append(obs)
        return obs, rewards, terminated, truncated, info

    def reset(self, *args, **kwargs):
        thing = super().reset(*args, **kwargs)
        self.obs_record = []
        return thing


def policy_wrapper(Policy: BaseAgentPolicy, agent_obs_normalizer, identity='wrapped_policy'):
    class WrappedPolicy(Policy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.identity = identity

        def set_team(self, team):
            if team == 0:
                self.team = Team.BLUE_TEAM
            else:
                self.team = Team.RED_TEAM

        def get_action(self, obs, *args, **kwargs):
            agent_obs = agent_obs_normalizer.unnormalized(obs)
            return self.compute_action({self.id: agent_obs})

    return WrappedPolicy
