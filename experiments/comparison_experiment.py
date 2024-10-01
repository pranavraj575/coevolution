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

    PARSER.add_argument('--MAP-Elites', action='store_true', required=False,
                        help="MAP-Elites for individual policy training")
    PARSER.add_argument('--MCAA-mainland', action='store_true', required=False,
                        help="MCAA mainland for team generation")

    PARSER.add_argument('--island-size', type=int, required=False, default=15,
                        help="population in each island")
    PARSER.add_argument('--games-per-epoch', type=int, required=False, default=16,
                        help="games to play per epoch"
                        )
    args = PARSER.parse_args()

    import torch, os, sys, ast, time, random, shutil
    import dill as pickle

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
    ident = (args.ident +
             thing +
             coevolution_string(args) +
             pyquaticus_string(args) +
             ('' if args.MCAA_mainland else berteam_string(args)) +
             ('_MAPE' if args.MAP_Elites else '') +
             ('_MCAA' if args.MCAA_mainland else '') +
             ('_gms_' + str(args.games_per_epoch)) +
             '_pop_' + str(args.island_size)
             )
    data_folder = os.path.join(DIR, 'data', 'temp', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)
    backup_dir = os.path.join(DIR, 'data', 'save', 'backups', ident)


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
    if args.MCAA_mainland:
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

    if args.MAP_Elites:
        TrainerClass = PZCC_MAPElites
        trainer_kwargs = {'default_behavior_radius': 1.}
    else:
        TrainerClass = ComparisionExperiment
        trainer_kwargs = dict()
    trainer = TrainerClass(population_sizes=pop_sizes,
                           games_per_epoch=args.games_per_epoch,
                           MCAA_mode=args.MCAA_mainland,
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
    exception_raised = False
    while True:
        try:
            plotting = {'init_dists': [],
                        'epochs': [],
                        }
            # if we dont want to reset, try a load if there was an exception
            if (exception_raised or (not args.reset)) and os.path.exists(save_dir):
                print('loading from', save_dir)
                trainer.load(save_dir=save_dir)
                f = open(os.path.join(save_dir, 'plotting.pkl'), 'rb')
                plotting = pickle.load(f)
                f.close()
                if exception_raised:
                    print("EXCEPTION CAUGHT, RELOADED FROM CHECKPOINT")
            print('seting save directory as', save_dir)


            def update_plotting_variables():
                if args.MCAA_mainland:
                    init_dist = trainer.team_trainer.distribution.detach().numpy()
                else:
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
                # TODO: past here we have not edited
                if idxs is None:
                    print('generated team (around the strongest) has elos', trainer.classic_elos[gen_team])

                    print('best agent has elo', trainer.classic_elos[best],
                          'and is type', typer(best))
                    print('worst agents has elo', trainer.classic_elos[worst],
                          'and is type', typer(worst))

                    print('playing generated (blue, ' + str(tuple((idx, typer(idx)) for idx in gen_team))
                          + ') against double best (red, ' + str(
                        tuple((best, typer(best)) for idx in range(team_size))) + ')')

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
                        if args.MCAA_mainland:
                            print('all pred win rates')
                            metric = trainer.elos.numpy()
                        else:
                            print('all ')
                            metric = trainer.classic_elos.numpy()

                        for identity in id_to_idxs:
                            print('\t', identity, 'agents:', metric[id_to_idxs[identity]])

                        print('avg ')
                        for identity in id_to_idxs:
                            print('\t', identity, 'agents:', np.mean(metric[id_to_idxs[identity]]))

                        print('max ')
                        for identity in id_to_idxs:
                            print('\t', identity, 'agents:', np.max(metric[id_to_idxs[identity]]))
                        if plotting['init_dists']:
                            print('initial values with metrics')
                            for prob, elo in zip(np.round(plotting['init_dists'][-1], 3), metric):
                                print('(', prob, ', ', round(elo, 2), ')', sep='', end=';\t')
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
                dist = trainer.team_trainer.get_total_distribution(T=team_size)
                f = open(os.path.join(save_dir, 'final_team_distribution.pkl'), 'wb')
                pickle.dump(dist, f)
                f.close()
            trainer.clear()

            shutil.rmtree(data_folder)
            break
        except Exception as e:
            exception_raised = True
            if str(e) == 'NAN found':
                print('Exception caught')
            else:
                raise Exception(e)
