import itertools

if __name__ == '__main__':
    import numpy as np
    import argparse
    from experiments.pyquaticus_utils.arg_parser import *

    PARSER = argparse.ArgumentParser()
    add_team_args(PARSER)
    add_learning_agent_args(PARSER, split_learners=True)
    add_coevolution_args(PARSER, clone_default=1)
    add_pyquaticus_args(PARSER)
    add_berteam_args(PARSER)
    add_experiment_args(PARSER, 'pyquaticus_coev_MLM')
    PARSER.add_argument('--dont-backup', action='store_true', required=False,
                        help="do not backup a copy of previous save")

    args = PARSER.parse_args()

    import torch, os, sys, ast, time, random, shutil
    import dill as pickle

    from unstable_baselines3 import WorkerPPO, WorkerDQN
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
                                  PositionalAppender)
    from BERTeam.buffer import BinnedReplayBufferDiskStorage
    from BERTeam.trainer import MLMTeamTrainer

    from src.coevolver import PettingZooCaptianCoevolution
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

    ppo_cnt = args.ppo_agents

    dqn_cnt = args.dqn_agents
    buffer_cap = args.replay_buffer_capacity
    net_arch = tuple(ast.literal_eval('(' + args.net_arch + ')'))

    lstm_dropout = args.lstm_dropout
    if lstm_dropout is None:
        lstm_dropout = args.dropout
    ident = (args.ident +
             '_agents_' +
             learning_agents_string(args) +
             coevolution_string(args) +
             pyquaticus_string(args) +
             berteam_string(args)
             )
    data_folder = os.path.join(DIR, 'data', 'temp', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)
    backup_dir = os.path.join(DIR, 'data', 'save', 'backups', ident)

    retrial_fn = lambda val: args.loss_retrials if val == 0 else (args.tie_retrials if val == .5 else 0)


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

    ppokwargs = dict()

    policy_kwargs = {
        'net_arch': dict(pi=net_arch,
                         vf=net_arch),
    }
    create_ppo = lambda i, env: (WorkerPPO(policy=PPOMlp,
                                           env=env,
                                           policy_kwargs={
                                               'net_arch': dict(pi=[64, 64],
                                                                vf=[64, 64]),
                                           },
                                           **ppokwargs
                                           ), train_info_dict.copy()
                                 )
    dqnkwargs = {
        'buffer_size': buffer_cap,
    }
    create_dqn = lambda i, env: (WorkerDQN(policy=DQNMlp,
                                           env=env,
                                           **dqnkwargs
                                           ), train_info_dict.copy()
                                 )

    if args.split_learners:
        pop_sizes = [ppo_cnt,
                     dqn_cnt,
                     ]

        worker_constructors = [create_ppo,
                               create_dqn,
                               ]
    else:
        pop_sizes = [ppo_cnt + dqn_cnt]


        def create_learner(i, env):
            if i < ppo_cnt:
                return create_ppo(i, env)
            else:
                return create_dqn(i - ppo_cnt, env)


        worker_constructors = [create_learner]
    if sum(pop_sizes) == 0:
        raise Exception("no agents specified, at least one of --*-agents must be greater than 0")
    max_cores = len(os.sched_getaffinity(0))
    if args.display:
        proc = 0
    else:
        proc = args.processes
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
    trainer = PettingZooCaptianCoevolution(population_sizes=pop_sizes,
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
                                           depth_to_retry_result=retrial_fn,
                                           # member_to_population=lambda team_idx, member_idx: {team_idx},
                                           # team_member_elo_update=1*np.log(10)/400,
                                           mutation_prob=args.mutation_prob,
                                           clone_replacements=clone_replacements,
                                           protect_elite=args.elite_protection,
                                           )
    plotting = {'init_dists': [],
                'team_dists': [],
                'team_dists_non_ordered': [],
                'epochs': [],
                }

    if not args.reset and os.path.exists(save_dir):
        print('loading from', save_dir)
        trainer.load(save_dir=save_dir)
        f = open(os.path.join(save_dir, 'plotting.pkl'), 'rb')
        plotting = pickle.load(f)
        f.close()
    print('seting save directory as', save_dir)


    def update_plotting_variables():
        test_team = trainer.team_trainer.create_masked_teams(T=team_size, N=1)
        indices, dist = trainer.team_trainer.get_member_distribution(init_team=test_team,
                                                                     indices=None,
                                                                     obs_preembed=None,
                                                                     obs_mask=None,
                                                                     noise_model=None,
                                                                     valid_members=None,
                                                                     )

        init_dist = torch.mean(dist, dim=0).detach().numpy()
        plotting['init_dists'].append(init_dist)
        # team_dist = team_trainer.get_total_distribution(T=team_size, N=1)
        # team_dist_non_ordered = dict()
        # for item in team_dist:
        #    key = tuple(sorted(item))
        #    if key not in team_dist_non_ordered:
        #        team_dist_non_ordered[key] = 0
        #    team_dist_non_ordered[key] += team_dist[item]
        # plotting['team_dists'].append(team_dist)
        # plotting['team_dists_non_ordered'].append(team_dist_non_ordered)
        plotting['epochs'].append(trainer.epochs)


    if args.display:
        test_animals = ([(RandPolicy(test_env.action_space(0)), non_train_dict.copy())] +
                        [(DefendPolicy(agent_id=0,
                                       team='red',
                                       mode=mode,
                                       flag_keepout=config_dict['flag_keepout'],
                                       catch_radius=config_dict["catch_radius"],
                                       using_pyquaticus=True,
                                       ), non_train_dict.copy()
                          )
                         for mode in ('easy', 'medium', 'hard')
                         ] +
                        [(AttackPolicy(agent_id=0,
                                       mode=mode,
                                       using_pyquaticus=True,
                                       ), non_train_dict.copy()
                          )
                         for mode in ('easy', 'medium', 'hard')
                         ]
                        )
    else:
        test_animals = []


    def typer(global_idx):
        if global_idx >= 0:
            animal, _ = trainer.load_animal(trainer.index_to_pop_index(global_idx))
        else:
            animal, _ = test_animals[global_idx]

        if isinstance(animal, WorkerPPO):
            return 'ppo'
        elif isinstance(animal, WorkerDQN):
            return 'dqn'
        elif 'WrappedPolicy' in str(type(animal)):
            return animal.identity + ' ' + animal.mode
        else:
            return 'rand'


    if args.display:
        idxs = args.idxs_to_display
        elos = trainer.classic_elos.numpy().copy()
        worst = np.argmin(elos)
        best = np.argmax(elos)

        elos[best] = -np.inf
        second_best = np.argmax(elos)

        classic_elos = trainer.classic_elos.numpy()
        idents_and_elos = []
        for i in range(sum(pop_sizes)):
            idents_and_elos.append((typer(i), classic_elos[i]))
        print('all elos by index')
        for i, animal in enumerate(test_animals):
            ip = i - len(test_animals)
            print(ip, ' (', typer(ip), '): ', None, sep='')
        for i, (identity, elo) in enumerate(idents_and_elos):
            print(i, ' (', identity, '): ', elo, sep='')
        env = env_constructor(None)
        gen_team = [t.item()
                    for t in trainer.create_team(team_idx=0,
                                                 captian=best,
                                                 obs_preembed=None,
                                                 obs_mask=None,
                                                 )[0].flatten()]
        if idxs is None:

            print('generated team (around the strongest) has elos', trainer.classic_elos[gen_team])

            print('best agent has elo', trainer.classic_elos[best],
                  'and is type', typer(best))
            print('worst agents has elo', trainer.classic_elos[worst],
                  'and is type', typer(worst))

            print('playing generated (blue, ' + str(tuple((idx, typer(idx)) for idx in gen_team))
                  + ') against double best (red, ' + str(tuple((best, typer(best)) for idx in range(team_size))) + ')')

            outcom = CTFOutcome()
            agents = []
            for team_idxs in gen_team, [best]*team_size:
                m = []
                for idx in team_idxs:
                    agent = trainer.load_animal(trainer.index_to_pop_index(idx))[0]
                    m.append(agent)
                agents.append(m)

            outcom.get_outcome(
                team_choices=[[torch.tensor(idx) for idx in gen_team], [torch.tensor(best)]*team_size],
                agent_choices=agents,
                env=env,
                updated_train_infos=[[non_train_dict]*team_size]*2,
            )

        else:
            teams = [ast.literal_eval('(' + team + ')') for team in idxs.split(';')]
            if len(teams) == 1:
                teams = [gen_team] + teams
            A, B = teams

            print('playing', A, '(blue, ' + str([typer(idx) for idx in A])
                  + ') against', B, '(red, ' + str([typer(idx) for idx in B]) + ')')

            outcom = CTFOutcome()
            agents = []
            for team_idxs in A, B:
                m = []
                for idx in team_idxs:
                    if idx >= 0:
                        agent = trainer.load_animal(trainer.index_to_pop_index(idx))[0]
                        m.append(agent)
                    else:
                        m.append(test_animals[idx][0])
                agents.append(m)

            outcom.get_outcome(
                team_choices=[[torch.tensor(a) for a in A], [torch.tensor(b) for b in B]],
                agent_choices=agents,
                env=env,
                updated_train_infos=[[non_train_dict]*team_size]*2,
            )
        if args.save_video is not None:
            print('saving video')
            env.write_video(video_file=args.save_video)
    else:
        while trainer.epochs < args.epochs:
            tim = time.time()
            print('starting epoch', trainer.info['epochs'], 'at time', time.strftime('%H:%M:%S'))
            if trainer.epochs == 0:
                update_plotting_variables()
            epoch_info = trainer.epoch(
                noise_model=team_trainer.create_nose_model_towards_uniform(
                    t=torch.exp(-np.log(2.)*trainer.ages/args.half_life)
                )
            )
            if not trainer.epochs%args.train_freq:
                print('training step started')
                trainer.team_trainer.train(
                    batch_size=args.batch_size,
                    minibatch=args.minibatch_size,
                    mask_probs=None,
                    replacement_probs=(.8, .1, .1),
                    mask_obs_prob=.1,
                )
                print('training step finished')
            if not trainer.epochs%args.train_freq:
                update_plotting_variables()

            if True:
                id_to_idxs = dict()
                for i in range(sum(pop_sizes)):
                    identity = typer(i)
                    if identity not in id_to_idxs:
                        id_to_idxs[identity] = []
                    id_to_idxs[identity].append(i)
                print('all elos')
                classic_elos = trainer.classic_elos.numpy()
                for identity in id_to_idxs:
                    print('\t', identity, 'agents:', classic_elos[id_to_idxs[identity]])

                print('avg elos')
                for identity in id_to_idxs:
                    print('\t', identity, 'agents:', np.mean(classic_elos[id_to_idxs[identity]]))

                print('max elos')
                for identity in id_to_idxs:
                    print('\t', identity, 'agents:', np.max(classic_elos[id_to_idxs[identity]]))
                if plotting['init_dists']:
                    print('initial dist')
                    print(np.round(plotting['init_dists'][-1], 3))
                    print('with elos')
                    for prob, elo in zip(np.round(plotting['init_dists'][-1], 3), classic_elos):
                        print('(', prob, ',', round(elo, 2), ')', sep='', end=';\t')
                    print()

            if not (trainer.info['epochs'])%args.ckpt_freq:
                if not args.dont_backup and os.path.exists(save_dir):
                    print('backing up')
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)
                    shutil.copytree(save_dir, backup_dir)
                    print('done backing up')
                print('saving model')
                trainer.save(save_dir)
                print('done saving model')
                print('saving ploting')
                f = open(os.path.join(save_dir, 'plotting.pkl'), 'wb')
                pickle.dump(plotting, f)
                f.close()
                print('done saving plotting')

            print('time', time.time() - tim)
            print()
        ###########################################################
        # analysis
        ###########################################################

        from experiments.pyquaticus_utils.dist_plot import deorder_total_dist
        from experiments.pyquaticus_utils.agent_aggression import all_agent_aggression, default_potential_opponents

        import matplotlib.pyplot as plt
        from pathos.multiprocessing import ProcessPool as Pool

        plt.rcParams.update({'font.size': 14})

        print("TRAINING FINISHED, now checking aggression for anal")
        classic_elos = trainer.classic_elos

        elo_trial_file = os.path.join(save_dir, 'elo_trials.pkl')
        elo_num_trial_file = os.path.join(save_dir, 'elo_num_trials.pkl')
        aggression_file = os.path.join(save_dir, 'aggression.pkl')
        total_dist_file = os.path.join(save_dir, 'total_dist.pkl')

        plot_dir = os.path.join(save_dir, 'plots')

        if os.path.exists(plot_dir):
            # clear any plots
            shutil.rmtree(plot_dir)
        os.makedirs(plot_dir)

        trial_nm = 10

        if os.path.exists(elo_num_trial_file):
            f = open(elo_num_trial_file, 'rb')
            completed_elo_trieals = pickle.load(f)
            f.close()
        else:
            completed_elo_trieals = dict()
        overall_completed_trials = 0


        def get_possible_teams(num):
            possible_teams = itertools.product(range(num), repeat=team_size)
            return tuple(set(
                tuple(sorted(team)) for team in possible_teams
            ))


        def get_outcome_trial(team):
            potential_opponents = default_potential_opponents(env_constructor=env_constructor)
            potential_opponent_teams = get_possible_teams(len(potential_opponents))
            outcome = CTFOutcome(quiet=True)
            env = env_constructor(None)
            info_dicts = [{DICT_TRAIN: False}
                          for _ in range(team_size)]
            info_dicts = info_dicts, info_dicts
            ret = dict()
            for opponent_team_idxs in potential_opponent_teams:
                opp_team = [potential_opponents[idx] for idx in opponent_team_idxs]
                come = outcome.get_outcome(team_choices=[[torch.tensor(0) for _ in range(team_size)],
                                                         [torch.tensor(-1) for _ in range(team_size)]],
                                           agent_choices=[team, opp_team],
                                           updated_train_infos=info_dicts,
                                           env=env,
                                           )
                ret[opponent_team_idxs] = {
                    'team result': come[0][0],
                    'opponent_ids': tuple(
                        opp.identity if 'identity' in dir(opp) else 'unknown'
                        for opp in opp_team
                    )
                }
            return ret


        while overall_completed_trials < trial_nm:
            print('running trials on all agent teams (could take forever)',
                  'at time', time.strftime('%H:%M:%S'))
            start = time.time()
            possible_teams = get_possible_teams(sum(pop_sizes))
            teams_to_inc = []
            left_to_inc = 0
            for team_idxs in possible_teams:
                if team_idxs not in completed_elo_trieals:
                    completed_elo_trieals[team_idxs] = 0

                if completed_elo_trieals[team_idxs] <= overall_completed_trials:
                    left_to_inc += 1
                    if len(teams_to_inc) >= max(1, proc*3):
                        # we dont want to do too many at a time
                        continue
                    teams_to_inc.append(team_idxs)
                    completed_elo_trieals[team_idxs] += 1
            print('on iter', overall_completed_trials, '; ',
                  'unfinished teams:', left_to_inc, '; ',
                  'teams this loop:', len(teams_to_inc)
                  )
            if not teams_to_inc:
                # in this case, we need to increase the overall counter,
                #   as we have done each possible team this many times
                overall_completed_trials += 1
                continue

            agents = (
                [trainer.load_animal(trainer.index_to_pop_index(idx), load_buffer=False)[0]
                 for idx in team_idxs]
                for team_idxs in teams_to_inc
            )
            if proc > 0:
                with Pool(processes=proc) as pool:
                    outcomes = pool.map(get_outcome_trial, agents)
            else:
                outcomes = [get_outcome_trial(agent) for agent in agents]

            print('finsihed at time', time.strftime('%H:%M:%S'))
            print('total time:', round(time.time() - start))

            elo_trials = dict()
            if os.path.exists(elo_trial_file):
                f = open(elo_trial_file, 'rb')
                elo_trials.update(pickle.load(f))
                f.close()

            for out_dict, team_idxs in zip(outcomes, teams_to_inc):
                if team_idxs not in elo_trials:
                    elo_trials[team_idxs] = []
                elo_trials[team_idxs].append(out_dict)

            print('saving outcomes')
            f = open(elo_trial_file, 'wb')
            pickle.dump(elo_trials, f)
            f.close()

            f = open(elo_num_trial_file, 'wb')
            pickle.dump(completed_elo_trieals, f)
            f.close()
            print('done saving')
            del elo_trials
            print(overall_completed_trials)

        if os.path.exists(aggression_file):
            f = open(aggression_file, 'rb')
            aggression = pickle.load(f)
            f.close()
        else:

            start = time.time()
            agents = (trainer.load_animal(trainer.index_to_pop_index(idx), load_buffer=False)[0]
                      for idx in range(sum(pop_sizes)))
            aggression = all_agent_aggression(agents=agents,
                                              env_constructor=env_constructor,
                                              team_size=team_size,
                                              processes=proc,
                                              )

            f = open(aggression_file, 'wb')
            pickle.dump(aggression, f)
            f.close()
            print('done with aggression, time:', round(time.time() - start))

        # divisions = None
        divisions = [1.]
        default_num_divisions = 3
        # divisions to use to divide aggresion
        # if divisions is None, does tertiles
        if divisions is None:
            test = np.array(sorted(aggression))
            divisions = [np.quantile(test, q).item() for q in
                         (i/default_num_divisions for i in range(1, default_num_divisions))
                         ]
        if os.path.exists(total_dist_file):
            f = open(total_dist_file, 'rb')
            original_total_dist = pickle.load(f)
            f.close()
        else:
            print('calculating total dist')
            start = time.time()
            original_total_dist = trainer.team_trainer.get_total_distribution(T=team_size,
                                                                              N=1,
                                                                              )
            print('done with total dist, time:', round(time.time() - start))
            f = open(total_dist_file, 'wb')
            pickle.dump(original_total_dist, f)
            f.close()

        for ordered in (True, False):
            if ordered:
                total_dist = original_total_dist
            else:
                total_dist = deorder_total_dist(original_total_dist)


            # analyze based on bins of 'aggressiveness'
            def idx_to_bin(idx):
                agg = aggression[idx]
                bin = 0
                while bin < len(divisions) and divisions[bin] < agg:
                    bin += 1
                return bin
                elo = classic_elos[idx].item()
                return str(bin) + ['b', 'g'][int(elo > 1000)]


            all_bins = list(set(idx_to_bin(idx) for idx in range(sum(pop_sizes))))


            def team_to_team_type(team):
                team_type = tuple(idx_to_bin(idx) for idx in team)
                if not ordered:
                    team_type = tuple(sorted(team_type))
                return team_type


            def avg_cosine_similarity(idxs):
                cos_sim = 0
                cnt = 0
                for choice in itertools.combinations(idxs, 2):
                    agenti_embed, agentj_embed = [team_trainer.team_builder.berteam.agent_embedding(torch.tensor(idx))
                                                  for idx in choice]
                    cos_sim += torch.nn.CosineSimilarity(0)(agenti_embed, agentj_embed).detach().item()
                    cnt += 1
                return cos_sim/cnt


            team_type_dist = dict()
            for team_idxs in total_dist:
                prob = total_dist[team_idxs]
                team_type = team_to_team_type(team_idxs)
                if team_type not in team_type_dist:
                    team_type_dist[team_type] = 0
                team_type_dist[team_type] += prob

            values = sorted(team_type_dist.keys(), key=lambda t: team_type_dist[t], reverse=True)

            ax = plt.gca()
            plt.bar(range(len(values)), [team_type_dist[t] for t in values])
            ax.set_xticks(range(len(values)), values)
            # plt.bar(range(4), [team_type_dist[t] for t in values[:4]])
            # ax.set_xticks(range(4), values[:4])
            plt.title(('' if ordered else 'Unordered ') + 'BERTeam Occurrence Probabilities')
            key = None
            if len(divisions) == 2:
                key = 'AGENT KEY:\n0: Defensive\n1: Balanced\n2: Aggressive'
            else:
                key = 'AGENT KEY:\n0: Defensive\n' + str(len(divisions)) + ': Aggressive'
            if key is not None:
                plt.text(.69, .4, key,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                         transform=ax.transAxes,
                         verticalalignment='center',
                         # horizontalalignment='center',
                         )
            plt.xlabel('Team Composition')
            plt.ylabel("Probability")
            plt.ylim(0, 1)
            plt.savefig(os.path.join(plot_dir, ('' if ordered else 'unordered_') + 'berteam_aggression_plot.png'),
                        bbox_inches='tight')
            plt.close()
            # plt.scatter(aggression, classic_elos)
            plt.ylabel('Evolution Elo')
            plt.xlabel('Aggression Metric')
            plt.title('Evolved Population Elos and Aggression')
            cos_similarity_rec = dict()
            for binnm in all_bins:
                agent_idxs = [idx for idx in range(sum(pop_sizes)) if idx_to_bin(idx) == binnm]
                # plt.ylabel('Elo')
                # plt.xlabel('Aggression Metric')
                if binnm == 0:
                    name = 'Defensive'
                elif binnm == len(divisions):
                    name = 'Aggressive'
                else:
                    name = 'Bin ' + str(binnm)
                plt.scatter([aggression[i] for i in agent_idxs],
                            classic_elos[agent_idxs],
                            label=name)
                if ordered:
                    avg_cos = avg_cosine_similarity(agent_idxs)
                    print('\nbin', binnm)
                    print('average cosine similarity:', avg_cos)
                    print('count', len(agent_idxs))
                    cos_similarity_rec[binnm] = avg_cos

            plt.legend()
            plt.savefig(os.path.join(plot_dir, 'aggression_with_elo.png'), bbox_inches='tight')
            plt.close()
            if ordered:
                avg_cos = avg_cosine_similarity(range(sum(pop_sizes)))
                print('overall avg cosine similarity:', avg_cos)

    trainer.clear()

    shutil.rmtree(data_folder)
