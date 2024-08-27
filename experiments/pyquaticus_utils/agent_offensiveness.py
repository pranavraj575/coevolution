import itertools

from unstable_baselines3.utils import DICT_TRAIN

from experiments.pyquaticus_utils.outcomes import CTFOutcome


def test_agent_offensiveness(agent,
                             env,
                             test_team,
                             info_dicts=None,
                             team_size=2,
                             ):
    """
    gets agent offensiveness by playing a game of (agent, agent,...) against test_teams

    Args:
        agent: agent being evaluated
        env: environment to evaluate in (DOES NOT RESET ENV AFTER)
        test_team: team to test against
        info_dicts: info dict to send to training alg
        team_size: size of teams
    Returns:
        offesniveness of agent
    """
    if info_dicts is None:
        info_dicts = [{DICT_TRAIN: False}
                      for _ in range(team_size)]
        info_dicts = info_dicts, info_dicts
    outcom = CTFOutcome()
    env.reset()

    agent_team = [agent for _ in range(team_size)]
    agent_idx = [0 for _ in range(team_size)]  # not important for CTFOutcome
    opponent_idx = [-1 for _ in range(team_size)]  # not important for CTFOutcome
    index_choices = [agent_idx, opponent_idx]
    offensiveness, _ = outcom.outcome_team_offensiveness(
        agent_choices=[agent_team, test_team],
        index_choices=index_choices,
        updated_train_infos=info_dicts,
        env=env,
    )
    return offensiveness


def agent_offensiveness(agent,
                        env,
                        potential_opponents,
                        info_dicts=None,
                        team_size=2,
                        ):
    """
    gets agent offensiveness by playing a game of (agent, agent,...) against all possible teams of fixed opponents
    Args:
        agent: agent being evaluated
        env: environment to evaluate in
        potential_opponents: possible opponent list
        info_dicts: info dict to send to training alg
        team_size: size of teams
    Returns:
        mean offesniveness of agent across all traials
    """
    # sort t, as we do not care about order
    test_teams = [tuple(sorted(t)) for t in itertools.product(potential_opponents, repeat=team_size)]
    test_teams = list(set(test_teams))  # remove duplicates
    all_offensiveness = []
    for opponent_team in test_teams:
        offensiveness = test_agent_offensiveness(agent=agent,
                                                 env=env,
                                                 test_team=opponent_team,
                                                 info_dicts=info_dicts,
                                                 team_size=team_size,
                                                 )
        all_offensiveness.append(offensiveness)
    return sum(all_offensiveness)/len(all_offensiveness)


if __name__ == '__main__':
    from experiments.pyquaticus_utils.wrappers import MyQuaticusEnv, policy_wrapper

    from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
    from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker
    from repos.pyquaticus.pyquaticus.config import config_dict_std

    config_dict = config_dict_std
    config_dict["max_screen_size"] = (float('inf'), float('inf'))

    team_size = 2
    test_env = MyQuaticusEnv(render_mode=None,
                             team_size=team_size,
                             config_dict=config_dict,
                             )

    obs_normalizer = test_env.agent_obs_normalizer
    obs_dim = obs_normalizer.flattened_length
    def_eazy = policy_wrapper(BaseDefender,
                              agent_obs_normalizer=obs_normalizer,
                              )(agent_id=0,
                                team='blue',
                                mode='easy',
                                flag_keepout=config_dict['flag_keepout'],
                                catch_radius=config_dict["catch_radius"],
                                using_pyquaticus=True,
                                )
    def_hard = policy_wrapper(BaseDefender,
                              agent_obs_normalizer=obs_normalizer,
                              )(agent_id=0,
                                team='blue',
                                mode='hard',
                                flag_keepout=config_dict['flag_keepout'],
                                catch_radius=config_dict["catch_radius"],
                                using_pyquaticus=True,
                                )
    att_easy = policy_wrapper(BaseAttacker,
                              agent_obs_normalizer=obs_normalizer,
                              )(agent_id=0,
                                mode='easy',
                                using_pyquaticus=True,
                                )
    att_hard = policy_wrapper(BaseAttacker,
                              agent_obs_normalizer=obs_normalizer,
                              )(agent_id=0,
                                mode='hard',
                                using_pyquaticus=True,
                                )

    off_att = test_agent_offensiveness(agent=att_hard,
                                       env=test_env,
                                       test_team=[att_easy, att_easy],
                                       )
    off_def = test_agent_offensiveness(agent=def_hard,
                                       env=test_env,
                                       test_team=[att_easy, att_easy],
                                       )
    print(off_att, off_def)
