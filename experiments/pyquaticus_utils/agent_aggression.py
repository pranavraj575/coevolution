import itertools
from pathos.multiprocessing import ProcessPool as Pool

from unstable_baselines3.utils import DICT_TRAIN

from experiments.pyquaticus_utils.outcomes import CTFOutcome
from experiments.pyquaticus_utils.wrappers import policy_wrapper
from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker
from experiments.pyquaticus_utils.reward_fns import RandPolicy


def test_agent_aggression(agent,
                          env,
                          test_team,
                          info_dicts=None,
                          team_size=2,
                          ):
    """
    gets agent aggression by playing a game of (agent, agent,...) against test_teams

    Args:
        agent: agent being evaluated
        env: environment to evaluate in (DOES NOT RESET ENV AFTER)
        test_team: team to test against
        info_dicts: info dict to send to training alg
        team_size: size of teams
    Returns:
        aggression of agent
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
    aggression, _ = outcom.outcome_team_aggressiveness(
        agent_choices=[agent_team, test_team],
        index_choices=index_choices,
        updated_train_infos=info_dicts,
        env=env,
    )
    return aggression


def agent_aggression(agent,
                     env,
                     potential_opponents,
                     info_dicts=None,
                     team_size=2,
                     ):
    """
    gets agent aggression by playing a game of (agent, agent,...) against all possible teams of fixed opponents
    Args:
        agent: agent being evaluated
        env: environment to evaluate in
        potential_opponents: possible opponent list
        info_dicts: info dict to send to training alg
        team_size: size of teams
    Returns:
        mean aggression of agent across all traials
    """
    potential_opponents = list(potential_opponents)
    # sort t, as we do not care about order
    test_teams = [tuple(sorted(t)) for t in itertools.product(range(len(potential_opponents)), repeat=team_size)]
    test_teams = list(set(test_teams))  # remove duplicates
    all_aggression = []
    for opponent_idxs in test_teams:
        opponent_team = [potential_opponents[idx] for idx in opponent_idxs]
        aggression = test_agent_aggression(agent=agent,
                                           env=env,
                                           test_team=opponent_team,
                                           info_dicts=info_dicts,
                                           team_size=team_size,
                                           )
        all_aggression.append(aggression)
    return sum(all_aggression)/len(all_aggression)


def default_potential_opponents(env_constructor):
    test_env = env_constructor(None)
    test_env.reset()
    config_dict = test_env.config_dict
    obs_normalizer = test_env.agent_obs_normalizer
    def_easy, def_mid, def_hard = [policy_wrapper(BaseDefender,
                                                  agent_obs_normalizer=obs_normalizer,
                                                  identity='def ' + mode
                                                  )(agent_id=0,
                                                    team='blue',
                                                    mode=mode,
                                                    flag_keepout=config_dict['flag_keepout'],
                                                    catch_radius=config_dict["catch_radius"],
                                                    using_pyquaticus=True,
                                                    )
                                   for mode in ('easy', 'medium', 'hard')
                                   ]

    att_easy, att_mid, att_hard = [policy_wrapper(BaseAttacker,
                                                  agent_obs_normalizer=obs_normalizer,
                                                  identity='att ' + mode
                                                  )(agent_id=0,
                                                    mode=mode,
                                                    using_pyquaticus=True,
                                                    )
                                   for mode in ('easy', 'medium', 'hard')
                                   ]
    rand = RandPolicy(test_env.action_space(0))
    potential_opponents = [att_easy, att_mid, att_hard,
                           def_easy, def_mid, def_hard,
                           rand,
                           ]
    return potential_opponents


def all_agent_aggression(agents, env_constructor, potential_opponents=None, team_size=2, processes=0):
    """

    Args:
        agents: generator of agents, each agent must be able to be a team member of outcome
        env_constructor:
        potential_opponents:
        team_size:
        processes:

    Returns:
        list of scalars that is eqch agents aggressiveness in order
    """
    if potential_opponents is None:
        potential_opponents = default_potential_opponents(env_constructor=env_constructor)

    def get_aggression(agent, train_infos=None):
        env = env_constructor(train_infos)
        aggressiveness = agent_aggression(agent=agent,
                                          env=env,
                                          potential_opponents=potential_opponents,
                                          team_size=team_size,
                                          )
        return aggressiveness

    if processes == 0:
        aggression = [get_aggression(agent) for agent in agents]
    else:
        with Pool(processes=processes) as pool:
            aggression = pool.map(get_aggression, agents)
    return aggression


print("WARNING: USING AGENT AGGRESSION, MUST CHANGE LINE 763 IN PYQUATICUS SO OBSTACLES DONT CHANGE SCORE")
# if player.team == Team.RED_TEAM:
#     self.game_score['blue_tags'] += 0
# else:
#   self.game_score['red_tags'] += 0

if __name__ == '__main__':
    import time

    from experiments.pyquaticus_utils.wrappers import MyQuaticusEnv

    from repos.pyquaticus.pyquaticus.config import config_dict_std

    config_dict = config_dict_std
    config_dict["max_screen_size"] = (float('inf'), float('inf'))
    config_dict['sim_speedup_factor'] = 1

    team_size = 2
    test_env = MyQuaticusEnv(render_mode=None,
                             team_size=team_size,
                             config_dict=config_dict,
                             )
    test_env.reset()

    obs_normalizer = test_env.agent_obs_normalizer
    obs_dim = obs_normalizer.flattened_length
    def_easy, def_mid, def_hard = [policy_wrapper(BaseDefender,
                                                  agent_obs_normalizer=obs_normalizer,
                                                  identity='def ' + mode
                                                  )(agent_id=0,
                                                    team='blue',
                                                    mode=mode,
                                                    flag_keepout=config_dict['flag_keepout'],
                                                    catch_radius=config_dict["catch_radius"],
                                                    using_pyquaticus=True,
                                                    )
                                   for mode in ('easy', 'medium', 'hard')
                                   ]

    att_easy, att_medium, att_hard = [policy_wrapper(BaseAttacker,
                                                     agent_obs_normalizer=obs_normalizer,
                                                     identity='att ' + mode
                                                     )(agent_id=0,
                                                       mode=mode,
                                                       using_pyquaticus=True,
                                                       )
                                      for mode in ('easy', 'medium', 'hard')
                                      ]
    rand = RandPolicy(test_env.action_space)

    env_cons = lambda train_infos: MyQuaticusEnv(render_mode=None,
                                                 team_size=team_size,
                                                 config_dict=config_dict,
                                                 )

    start = time.time()
    agents = [def_easy, def_mid, def_hard, att_easy, att_medium, att_hard]
    aggression = all_agent_aggression(agents=(a for a in agents),
                                      env_constructor=env_cons,
                                      potential_opponents=[def_easy,
                                                           def_mid,
                                                           def_hard,
                                                           att_easy,
                                                           att_medium,
                                                           att_hard,
                                                           ],
                                      team_size=team_size,
                                      processes=6,
                                      )
    for agent, off in zip(agents, aggression):
        print(agent.identity, ':', off)
    print('time taken:', round(time.time() - start))
    quit()
    for agent in (def_easy, def_mid, def_hard,
                  att_easy, att_medium, att_hard):
        start = time.time()
        print('testing')
        print(agent.identity)
        aggression = agent_offensiveness(agent=agent,
                                         env=test_env,
                                         potential_opponents=[def_easy, def_mid, def_hard,
                                                              att_easy, att_medium, att_hard],
                                         team_size=team_size
                                         )
        print(aggression)
        print('time taken:', round(time.time() - start))
