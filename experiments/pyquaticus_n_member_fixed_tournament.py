if __name__ == '__main__':
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--team-size', type=int, required=False, default=2,
                        help="size of teams")
    PARSER.add_argument('--samples', type=int, required=False, default=5,
                        help="samples to take")
    PARSER.add_argument('--arena-size', action='store', required=False, default='200.0,100.0',
                        help="x,y dims of arena, in format '200.0,100.0'")
    PARSER.add_argument('--processes', type=int, required=False, default=0,
                        help="number of processes to use")
    args = PARSER.parse_args()
    import torch, os, sys, itertools, time, ast, numpy as np, dill as pickle

    from multiprocessing import Pool

    from repos.pyquaticus.pyquaticus.config import config_dict_std
    from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
    from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker

    from experiments.pyquaticus_utils.reward_fns import RandPolicy
    from experiments.pyquaticus_utils.wrappers import MyQuaticusEnv, policy_wrapper
    from experiments.pyquaticus_utils.outcomes import CTFOutcome

    from src.utils.dict_keys import *

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    save_file = 'torunament' + ('_team_size_' + str(args.team_size) +
                                '_arena_size_' + args.arena_size.replace('.', '_').replace(',', '__'))

    save_dir = os.path.join(DIR, 'data', 'save', 'basic_team_tournament', save_file + '.pkl')
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    print('saving to', save_dir)


    def keify(thing):
        return tuple(sorted(thing))


    num_ags = 7
    possible_teams = sorted(
        {tuple(sorted(t)) for t in itertools.product(*[range(num_ags) for _ in range(args.team_size)])})

    team_to_idx = {t: i for i, t in enumerate(possible_teams)}

    result_dict = {
        t1: {
            t2: [0, 0, 0]
            for t2 in possible_teams if t2 != t1
        }
        for t1 in possible_teams
    }
    total = 0

    if os.path.exists(save_dir):
        f = open(save_dir, 'rb')
        result_dict = pickle.load(f)
        f.close()
        for a in result_dict:
            for b in result_dict[a]:
                total = sum(result_dict[a][b])
                break
            break
        if total >= args.samples:

            elo_conversion = 400/np.log(10)
            elo_update = 32*np.log(10)/400
            print('total value', total, 'is more than num samples', args.samples)
            temp_possible_teams = possible_teams
            # temp_possible_teams = [t for t in possible_teams if 2 in t]
            for tie_counts in True, False:
                print()
                print('tie counts:', tie_counts)
                f = open(save_dir, 'rb')
                result_dict = pickle.load(f)
                f.close()
                elos = torch.zeros(len(possible_teams))
                for _ in range(100):
                    old_elos = elos.clone()
                    for (A, B) in itertools.combinations(temp_possible_teams, 2):
                        w, t, l = result_dict[A][B]

                        if not tie_counts and (w > 0 or l > 0):
                            t = 0
                        expectation = torch.softmax(elos[(team_to_idx[A], team_to_idx[B]),], dim=-1)
                        actual = torch.tensor([w + t/2, l + t/2])/(w + t + l)
                        elos[(team_to_idx[A], team_to_idx[B]),] += elo_update*(actual - expectation)
                    if torch.linalg.norm(elos - old_elos) == 0:
                        break
                converted = elos*elo_conversion + 1000
                win_prob = torch.softmax(elos, -1)
                temp_possible_teams.sort(key=lambda team: elos[team_to_idx[team]].item())
                for team in temp_possible_teams:
                    idx = team_to_idx[team]
                    print(team, ':', converted[idx].item(), ':', win_prob[idx].item())
            quit()
    team_size = args.team_size

    config_dict = config_dict_std
    config_dict["max_screen_size"] = (float('inf'), float('inf'))
    config_dict["sim_speedup_factor"] = 10
    config_dict["max_time"] = 420.
    config_dict["world_size"] = ast.literal_eval('(' + args.arena_size + ')')
    test_env = MyQuaticusEnv(render_mode=None,
                             team_size=team_size,
                             config_dict=config_dict,
                             )

    obs_normalizer = test_env.agent_obs_normalizer
    obs_dim = obs_normalizer.flattened_length
    policies = dict()
    modes = ['easy', 'medium', 'hard']
    for key, Class in (('def', BaseDefender),
                       ('att', BaseAttacker),
                       ):
        policies[key] = []
        for mode in modes:
            policies[key].append(
                policy_wrapper(Class,
                               agent_obs_normalizer=obs_normalizer,
                               identity=key + ' ' + mode,
                               )
            )



    def env_constructor(train_infos):
        return MyQuaticusEnv(render_mode=None,
                             team_size=team_size,
                             config_dict=config_dict,
                             )


    non_train_dict = {DICT_TRAIN: False,
                      DICT_CLONABLE: False,
                      DICT_CLONE_REPLACABLE: False,
                      DICT_MUTATION_REPLACABLE: False,
                      DICT_IS_WORKER: False,
                      }
    all_agents = []
    for i in range(3):
        all_agents.append(policies['att'][i](agent_id=0,
                                             mode=modes[i%len(modes)],
                                             using_pyquaticus=True,
                                             )
                          )
    for i in range(3):
        all_agents.append(policies['def'][i](agent_id=0,
                                             team='red',
                                             mode=modes[i%len(modes)],
                                             flag_keepout=config_dict['flag_keepout'],
                                             catch_radius=config_dict["catch_radius"],
                                             using_pyquaticus=True,
                                             )
                          )
    all_agents.append(RandPolicy(test_env.action_space(0)))

    outcom = CTFOutcome()
    outcom.set_dir(os.path.join(DIR, 'data', 'temp', 'trash'))


    def evaluate(teams):
        A, B = teams
        for team in A, B:
            m = []
            for idx in team:
                agent = all_agents[idx]
                m.append(agent)
            agents.append(m)

        (ascore, _), (bscore, _) = outcom.get_outcome(
            team_choices=[[torch.tensor(m) for m in team] for team in (A, B)],
            agent_choices=agents,
            env=env_constructor(None),
            updated_train_infos=[[non_train_dict]*team_size]*2,
        )
        return ascore, bscore


    tim = time.time()
    for i in range(args.samples - total):
        pairs = []
        for A, B in itertools.combinations(possible_teams, 2):
            agents = []
            stim = time.time()
            if torch.rand(1) < .5:
                A = A[::-1]
            if torch.rand(1) < .5:
                B = B[::-1]

            if torch.rand(1) < .5:
                A, B = B, A
            pairs.append((A, B))
        if args.processes == 0:
            results = [evaluate(teams=teams) for teams in pairs]
        else:
            print('starting pool')
            with Pool(processes=args.processes) as pool:
                results = pool.map(evaluate, pairs)

        for (A, B), (ascore, bscore) in zip(pairs, results):
            A = keify(A)
            B = keify(B)
            # result_dict[A][B] is a win/loss/tie count for A vs B (win means A won)
            if ascore == bscore:
                print(A, 'tied with', B)
                result_dict[A][B][1] += 1
                result_dict[B][A][1] += 1
            if ascore > bscore:
                print(A, 'won agnst', B)
                # a won
                result_dict[A][B][0] += 1
                result_dict[B][A][2] += 1
            if ascore < bscore:
                print(B, 'won agnst', A)
                # b won
                result_dict[A][B][2] += 1
                result_dict[B][A][0] += 1
        print('total time:', round(time.time() - tim))
        f = open(save_dir, 'wb')
        pickle.dump(result_dict, f)
        f.close()
