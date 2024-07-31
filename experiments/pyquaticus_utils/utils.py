from repos.pyquaticus.pyquaticus.base_policies.base import BaseAgentPolicy


def policy_wrapper(Policy: BaseAgentPolicy):
    class WrappedPolicy(Policy):
        def get_action(self, obs, *args, **kwargs):
            return self.compute_action(obs)

    return WrappedPolicy
