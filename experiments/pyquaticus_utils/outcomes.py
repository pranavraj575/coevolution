import torch

from repos.pyquaticus.pyquaticus.base_policies.base import BaseAgentPolicy
from repos.pyquaticus.pyquaticus.structs import Team
from src.game_outcome import PlayerInfo, PettingZooOutcomeFn
from experiments.pyquaticus_utils.wrappers import MyQuaticusEnv

from unstable_baselines3.common.auto_multi_alg import AutoMultiAgentAlgorithm


class CTFOutcome(PettingZooOutcomeFn):
    def _get_outcome_from_agents(self, agent_choices, index_choices, updated_train_infos, env):
        updated_train_infos = updated_train_infos[0] + updated_train_infos[1]
        for team_idx, team in enumerate(agent_choices):
            for agent in team:
                if isinstance(agent, BaseAgentPolicy):
                    # tell the agent which team it is on
                    # print('skipping team assignment to see what happens')
                    agent.team = (Team.BLUE_TEAM, Team.RED_TEAM)[team_idx]
        agent_choices = agent_choices[0] + agent_choices[1]

        # env is set up so the first k agents are team blue and the last k agents are team red
        alg = AutoMultiAgentAlgorithm(env=env,
                                      workers={i: agent_choices[i] for i in range(len(agent_choices))},
                                      worker_infos={i: updated_train_infos[i] for i in range(len(agent_choices))},
                                      )
        alg.learn(total_timesteps=10000,
                  number_of_eps=1,
                  )
        score = (env.unwrapped.game_score['blue_captures'], env.unwrapped.game_score['red_captures'])

        if isinstance(env, MyQuaticusEnv):
            # get list of observations from each player
            team_0_obs = (torch.stack([torch.tensor(obs[agent_idx]) for obs in env.obs_record], dim=0)
                          for agent_idx in range(0, env.team_size))
            team_1_obs = (torch.stack([torch.tensor(obs[agent_idx]) for obs in env.obs_record], dim=0)
                          for agent_idx in range(env.team_size, 2*env.team_size))

            team_0_player_infos = [PlayerInfo(obs_preembed=episode_obs) for episode_obs in team_0_obs]
            team_1_player_infos = [PlayerInfo(obs_preembed=episode_obs) for episode_obs in team_1_obs]
        else:
            team_0_player_infos = []
            team_1_player_infos = []

        if score[0] == score[1]:
            return [
                (.5, team_0_player_infos),
                (.5, team_1_player_infos),
            ]
        if score[0] > score[1]:
            return [
                (1, team_0_player_infos),
                (0, team_1_player_infos),
            ]
        if score[0] < score[1]:
            return [
                (0, team_0_player_infos),
                (1, team_1_player_infos),
            ]
