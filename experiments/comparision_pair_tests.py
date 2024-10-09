import itertools

if __name__ == '__main__':
    import numpy as np
    import argparse
    from experiments.pyquaticus_utils.arg_parser import *

    PARSER = argparse.ArgumentParser()
    add_team_args(PARSER)
    PARSER.add_argument('--replay-buffer-capacity', type=int, required=False, default=10000,
                        help="replay buffer capacity")
    PARSER.add_argument('--net-arch', action='store', required=False, default='64,64',
                        help="hidden layers of network, should be readable as a list or tuple")
    PARSER.add_argument('--net-arch2', action='store', required=False, default='96,96',
                        help="hidden layers of network, should be readable as a list or tuple")
    add_coevolution_args(PARSER, clone_default=1)
    add_pyquaticus_args(PARSER)
    add_berteam_args(PARSER)
    add_experiment_args(PARSER, 'pyq_comp_exp')
    PARSER.add_argument('--dont-backup', action='store_true', required=False,
                        help="do not backup a copy of previous save")

    PARSER.add_argument('--island-size', type=int, required=False, default=15,
                        help="population in each island")
    PARSER.add_argument('--games-per-epoch', type=int, required=False, default=16,
                        help="games to play per epoch"
                        )
    PARSER.add_argument('--sample-games', type=int, required=False, default=50,
                        help="sample games to test different algorithms with")
    args = PARSER.parse_args()

    import torch, os, sys, ast, time, random, shutil, pickle
    from pathos.multiprocessing import ProcessPool as Pool

    from unstable_baselines3 import WorkerPPO, WorkerDQN, WorkerA2C
    from unstable_baselines3.dqn import MlpPolicy as DQNMlp
    from unstable_baselines3.ppo import MlpPolicy as PPOMlp

    if not args.unblock_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from repos.pyquaticus.pyquaticus.config import config_dict_std
    from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
    from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker

    from experiments.pyquaticus_utils.reward_fns import custom_rew, custom_rew2, RandPolicy
    from experiments.pyquaticus_utils.wrappers import MyQuaticusEnv, policy_wrapper
    from experiments.pyquaticus_utils.outcomes import CTFOutcome

    from BERTeam.networks import (BERTeam,
                                  TeamBuilder,
                                  LSTEmbedding,
                                  IdentityEncoding,
                                  ClassicPositionalEncoding,
                                  PositionalAppender,
                                  )
    from BERTeam.buffer import BinnedReplayBufferDiskStorage
    from BERTeam.trainer import MLMTeamTrainer
    from experiments.subclasses.MCAA_mainland_team_trainer import MCAAMainland
    from experiments.subclasses.coevolution_subclass import ComparisionExperiment, PZCC_MAPElites
    from experiments.pyquaticus_utils.agent_aggression import default_potential_opponents

    from src.utils.dict_keys import *

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    team_size = args.team_size

    reward_config = {i: custom_rew2 for i in range(2*team_size)}  # Example Reward Config

    config_dict = config_dict_std
    update_config_dict(config_dict, args)

    arena_size = ast.literal_eval('(' + args.arena_size + ')')

    test_env = MyQuaticusEnv(render_mode=None,
                             team_size=team_size,
                             config_dict=config_dict,
                             )

    obs_normalizer = test_env.agent_obs_normalizer
    DefendPolicy = policy_wrapper(BaseDefender, agent_obs_normalizer=obs_normalizer, identity='def')
    AttackPolicy = policy_wrapper(BaseAttacker, agent_obs_normalizer=obs_normalizer, identity='att')

    obs_dim = obs_normalizer.flattened_length

    RENDER_MODE = get_render_mode(args)

    clone_replacements = args.clone_replacements

    buffer_cap = args.replay_buffer_capacity
    net_arch = list(ast.literal_eval('(' + args.net_arch + ')'))
    net_arch2 = list(ast.literal_eval('(' + args.net_arch2 + ')'))

    lstm_dropout = args.lstm_dropout
    if lstm_dropout is None:
        lstm_dropout = args.dropout
    thing = (('_rb_cap_' + str(args.replay_buffer_capacity)) +
             ('_archs_' +
              ('_'.join([str(s) for s in net_arch])) + '__' +
              ('_'.join([str(s) for s in net_arch2]))
              )
             )

    overall_ident = ('COM_' +
                     args.ident +
                     thing +
                     coevolution_string(args) +
                     pyquaticus_string(args) +
                     berteam_string(args) +
                     ('_gms_' + str(args.games_per_epoch)) +
                     '_pop_' + str(args.island_size)
                     )
    overall_save_dir = os.path.join(DIR, 'data', 'save', overall_ident)
    if not os.path.exists(overall_save_dir):
        os.makedirs(overall_save_dir)
    data_file = os.path.join(overall_save_dir, 'data.pkl')


    def env_constructor(train_infos):
        return MyQuaticusEnv(save_video=args.save_video is not None,
                             render_mode=RENDER_MODE,
                             reward_config=reward_config,
                             team_size=team_size,
                             config_dict=config_dict,
                             frame_freq=args.frame_freq,
                             )


    non_train_dict = {DICT_TRAIN: False,
                      DICT_CLONABLE: False,
                      DICT_CLONE_REPLACABLE: False,
                      DICT_MUTATION_REPLACABLE: False,
                      DICT_IS_WORKER: False,
                      }
    train_info_dict = {DICT_TRAIN: True,
                       DICT_CLONABLE: True,
                       DICT_CLONE_REPLACABLE: True,
                       DICT_MUTATION_REPLACABLE: True,
                       DICT_IS_WORKER: True,
                       }

    ppo_policy_kwargs = {
        'net_arch': dict(pi=net_arch,
                         vf=net_arch),
    }

    ppo_policy_kwargs2 = {
        'net_arch': dict(pi=net_arch2,
                         vf=net_arch2),
    }

    dqn_policy_kwargs = {
        'net_arch': net_arch,
    }

    dqn_policy_kwargs2 = {
        'net_arch': net_arch2,
    }

    ppokwargs = dict()
    create_ppo = lambda i, env: (WorkerPPO(policy=PPOMlp,
                                           env=env,
                                           policy_kwargs=ppo_policy_kwargs,
                                           **ppokwargs
                                           ), train_info_dict.copy()
                                 )

    create_ppo2 = lambda i, env: (WorkerPPO(policy=PPOMlp,
                                            env=env,
                                            policy_kwargs=ppo_policy_kwargs2,
                                            **ppokwargs
                                            ), train_info_dict.copy()
                                  )
    dqnkwargs = {
        'buffer_size': buffer_cap,
    }
    create_dqn = lambda i, env: (WorkerDQN(policy=DQNMlp,
                                           env=env,
                                           policy_kwargs=dqn_policy_kwargs,
                                           **dqnkwargs
                                           ), train_info_dict.copy()
                                 )
    create_dqn2 = lambda i, env: (WorkerDQN(policy=DQNMlp,
                                            env=env,
                                            policy_kwargs=dqn_policy_kwargs2,
                                            **dqnkwargs
                                            ), train_info_dict.copy()
                                  )

    worker_constructors = [create_ppo,
                           create_ppo2,
                           create_dqn,
                           create_dqn2,
                           ]

    pop_sizes = [args.island_size for _ in worker_constructors]

    max_cores = len(os.sched_getaffinity(0))
    if args.display:
        proc = 0
    else:
        proc = args.processes
    algorithms = list(itertools.product((True, False), repeat=2))


    def sample_teams(MCAA, MAP_elites, n=1):
        ident = (args.ident +
                 thing +
                 coevolution_string(args) +
                 pyquaticus_string(args) +
                 ('' if MCAA else berteam_string(args)) +
                 ('_MAPE' if MAP_elites else '') +
                 ('_MCAA' if MCAA else '') +
                 ('_gms_' + str(args.games_per_epoch)) +
                 '_pop_' + str(args.island_size)
                 )
        data_folder = os.path.join(DIR, 'data', 'temp', ident)
        save_dir = os.path.join(DIR, 'data', 'save', ident)
        if not os.path.exists(save_dir):
            print("NO FILE FOUND", save_dir)
            return None

        if MCAA:
            team_trainer = MCAAMainland(pop_sizes=pop_sizes,
                                        )
        else:
            team_trainer = MLMTeamTrainer(
                team_builder=TeamBuilder(
                    input_embedder=LSTEmbedding(
                        input_dim=obs_dim,
                        embedding_dim=args.embedding_dim,
                        layers=args.lstm_layers,
                        dropout=lstm_dropout,
                        device=None,
                    ),
                    berteam=BERTeam(
                        num_agents=sum(pop_sizes),
                        embedding_dim=args.embedding_dim,
                        nhead=args.heads,
                        num_encoder_layers=args.encoders,
                        num_decoder_layers=args.decoders,
                        dim_feedforward=None,
                        dropout=args.dropout,
                        PosEncConstructor=PositionalAppender,
                    )
                ),
                buffer=BinnedReplayBufferDiskStorage(
                    capacity=args.capacity,
                    device=None,
                    bounds=[1/2, 1],
                )
            )

        if MAP_elites:
            TrainerClass = PZCC_MAPElites
            trainer_kwargs = {'default_behavior_radius': 1.}
        else:
            TrainerClass = ComparisionExperiment
            trainer_kwargs = dict()
        trainer = TrainerClass(population_sizes=pop_sizes,
                               games_per_epoch=args.games_per_epoch,
                               MCAA_mode=MCAA,
                               MCAA_fitness_update=.01,
                               team_trainer=team_trainer,
                               outcome_fn_gen=CTFOutcome,
                               env_constructor=env_constructor,
                               worker_constructors=worker_constructors,
                               storage_dir=data_folder,
                               protect_new=args.protect_new,
                               processes=proc,
                               # for some reason this overesetimates by a factor of 3, so we fix it
                               max_steps_per_ep=(
                                       1 + (1/3)*config_dict['render_fps']*config_dict['max_time']/
                                       config_dict['sim_speedup_factor']),
                               team_sizes=(team_size, team_size),
                               # member_to_population=lambda team_idx, member_idx: {team_idx},
                               # team_member_elo_update=1*np.log(10)/400,
                               mutation_prob=args.mutation_prob,
                               clone_replacements=clone_replacements,
                               protect_elite=args.elite_protection,
                               **trainer_kwargs,
                               )
        trainer.load(save_dir=save_dir)
        dist = trainer.team_trainer.get_total_distribution(T=2)
        keys = list(dist.keys())
        teams = []
        agents = []
        for _ in range(n):
            r = np.random.rand()
            i = -1
            while r > 0:
                i += 1
                r -= dist[keys[i]]
            team = keys[i]
            teams.append(team)
            agents.append([
                trainer.load_animal(trainer.index_to_pop_index(t), load_buffer=False)[0]
                for t in team
            ])

        return teams, agents


    def get_results(agent_choices):
        outcome = CTFOutcome(quiet=True)
        env = env_constructor(None)
        info_dicts = [{DICT_TRAIN: False}
                      for _ in range(team_size)]
        info_dicts = info_dicts, info_dicts
        flip = (np.random.rand() < .5)
        if flip:
            agent_choices = agent_choices[::-1]
        come = outcome.get_outcome(team_choices=[[torch.tensor(0) for _ in range(team_size)],
                                                 [torch.tensor(-1) for _ in range(team_size)]],
                                   agent_choices=agent_choices,
                                   updated_train_infos=info_dicts,
                                   env=env,
                                   )
        ((team_1_score, _), (team_2_score, _)) = come

        if flip:
            return team_2_score, team_1_score
        else:
            return team_1_score, team_2_score


    if (not args.reset) and os.path.exists(data_file):
        f = open(data_file, 'rb')
        stuff = pickle.load(f)
        f.close()
    else:
        stuff = dict()
    for alg1, alg2 in itertools.combinations(algorithms, 2):
        # key to save into results
        key = tuple(zip(('MCAA', "MAP Elites"), alg1)), tuple(zip(('MCAA', "MAP Elites"), alg2))
        if key not in stuff:
            stuff[key] = []
        old_len = len(stuff[key])
        todo = args.sample_games - old_len
        if todo <= 0:
            continue
        t1 = sample_teams(*alg1, n=todo)
        if t1 is None:
            continue
        t2 = sample_teams(*alg2, n=todo)
        if t2 is None:
            continue
        print('playing', todo, 'games for key', key)
        start = time.time()
        _, agents1 = t1
        _, agents2 = t2
        setups = zip(agents1, agents2)
        if proc > 0:
            with Pool(processes=proc) as pool:
                all_results = pool.map(get_results, setups)
        else:
            all_results = [get_results(agent_choices)
                           for agent_choices in setups]
        for res in all_results:
            stuff[key].append(res)
        print('time:', time.time() - start)
        print('saving...\r', end='\n')
        f = open(data_file, 'wb')
        pickle.dump(stuff, f)
        f.close()
        print('done saving')

    ####
    # base elos on fixed policy teams
    ####

    save_file_name = 'torunament' + ('_team_size_' + str(args.team_size) +
                                     '_arena_' + str('__'.join([str(t).replace('.', '_') for t in arena_size]))
                                     )


    def get_possible_teams(num):
        possible_teams = itertools.product(range(num), repeat=team_size)
        return tuple(set(
            tuple(sorted(team)) for team in possible_teams
        ))


    potential_opponents = default_potential_opponents(env_constructor=env_constructor)
    potential_opponent_teams = get_possible_teams(len(potential_opponents))
    for alg in algorithms:
        for opponent_team_idxs in potential_opponent_teams:
            opp_team = [potential_opponents[idx] for idx in opponent_team_idxs]
            key = tuple(zip(('MCAA', "MAP Elites"), alg)), ('against fixed pol team:', opponent_team_idxs)
            if key not in stuff:
                stuff[key] = []
            old_len = len(stuff[key])
            todo = args.sample_games - old_len
            if todo <= 0:
                continue
            t = sample_teams(*alg, n=todo)
            if t is None:
                continue
            _, agents = t
            print('playing', todo, 'games for key', key)
            start = time.time()

            setups = ([agent_choices_one_team, opp_team] for agent_choices_one_team in agents)
            if proc > 0:
                with Pool(processes=proc) as pool:
                    all_results = pool.map(get_results, setups)
            else:
                all_results = [get_results(agent_choices)
                               for agent_choices in setups]
            for res in all_results:
                stuff[key].append(res)
            print('time:', time.time() - start)
            print('saving...\r', end='\n')
            f = open(data_file, 'wb')
            pickle.dump(stuff, f)
            f.close()
            print('done saving')

    for key in stuff:
        if stuff[key]:
            print()
            print(key)
            s = np.zeros(2)
            for item in stuff[key]:
                s += item
            s /= len(stuff[key])
            print(s)
    basic_team_elo_file = os.path.join(DIR, 'data', 'save', 'basic_team_tournament', 'elos_' + save_file_name + '.pkl')
    f = open(basic_team_elo_file, 'rb')
    basic_team_to_elos = pickle.load(f)
    f.close()

    elos = torch.nn.Parameter(torch.zeros(len(algorithms)))
    alg_to_idx = {alg: i for i, alg in enumerate(algorithms)}
    optim = torch.optim.Adam(params=torch.nn.ParameterList([elos]),
                             lr=1e-2)


    def team_key_to_alg(team_key):
        return team_key[0][1], team_key[1][1]


    keys = list(stuff.keys())
    while True:
        old_elos = elos.data.clone()
        for i in torch.randperm(len(keys)):
            key = keys[i]
            team1_key, team2_key = key
            team1 = team_key_to_alg(team1_key)
            if not stuff[key]:
                continue
            optim.zero_grad()
            elo1 = elos[alg_to_idx[team1]]
            if type(team2_key[0]) == str:
                elo2 = torch.tensor(basic_team_to_elos[team2_key[1]])
            else:
                team2 = team_key_to_alg(team2_key)
                elo2 = elos[alg_to_idx[team2]]
            expectation = torch.softmax(torch.stack((elo1, elo2)), dim=-1)
            actual = torch.zeros(2)
            for item in stuff[key]:
                actual += torch.tensor(item)
            actual = actual/len(stuff[key])
            loss = torch.nn.MSELoss().forward(expectation, actual)
            loss.backward()
            optim.step()
        elo_diff = elos.data - old_elos
        if torch.linalg.norm(elo_diff) < .01:
            break

    elo_conversion = 400/np.log(10)
    print('mcaa, mape')
    print(algorithms)
    print(elos.data.detach().numpy()*elo_conversion + 1000)
    print(basic_team_to_elos[(2, 5)]*elo_conversion + 1000)
