# this one uses comparison experiment and generates captians according to distribution with noise
import argparse
import copy

if __name__ == '__main__':
    import numpy as np
    from experiments.pyquaticus_utils.arg_parser import *

    PARSER = argparse.ArgumentParser()
    add_team_args(PARSER)
    PARSER.add_argument('--rand-count', type=int, required=False, default=0,
                        help="number of random agents to add into population to try confusing team selector")
    PARSER.add_argument('--games-per-epoch', type=int, required=False, default=25,
                        help="guess")
    add_pyquaticus_args(PARSER, arena_size=True)
    add_berteam_args(PARSER)
    add_experiment_args(PARSER, 'basic_team_MLM2', epochs_default=2000)

    PARSER.add_argument('--plot', action='store_true', required=False,
                        help="skip training and plot")
    PARSER.add_argument('--cutoff', type=int, required=False, default=10,
                        help="number of teams to list in total distribution before cutting off")
    PARSER.add_argument('--dist-sample', type=int, required=False, default=30,
                        help="number of times to sample distribution for plotting (batch norm does not behave well with model.eval())")
    PARSER.add_argument('--smoothing', type=int, required=False, default=2,
                        help="smoothing to include in plot")
    PARSER.add_argument('--dont-backup', action='store_true', required=False,
                        help="do not backup a copy of previous save")
    # can stabilize training, equivalent to messing with weight of data
    PARSER.add_argument('-lr', '--learning-rate', type=float, required=False, default=None,
                        help="learning rate (adam default is .001)")
    args = PARSER.parse_args()

    import torch, os, sys, ast, time, shutil
    import dill as pickle

    if not args.unblock_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from repos.pyquaticus.pyquaticus.config import config_dict_std
    from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
    from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker

    from experiments.pyquaticus_utils.reward_fns import RandPolicy
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

    from experiments.subclasses.coevolution_subclass import ComparisionExperiment
    from src.utils.dict_keys import *

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    team_size = args.team_size
    lr = args.learning_rate
    config_dict = config_dict_std
    update_config_dict(config_dict, args)
    arena_size = ast.literal_eval('(' + args.arena_size + ')')
    arena_size = tuple(float(t) for t in arena_size)
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

    RENDER_MODE = get_render_mode(args)

    lstm_dropout = args.lstm_dropout
    if lstm_dropout is None:
        lstm_dropout = args.dropout
    if lr is not None:
        optim_kwargs = {'lr': lr}
        lr_str = '_lr_' + str(lr).replace('.', '_')
    else:
        optim_kwargs = None
        lr_str = ''
    rand_cnt = args.rand_count
    ident = (args.ident +
             lr_str +
             pyquaticus_string(args) +
             berteam_string(args) +
             (('_rand_cnt_' + str(rand_cnt)) if rand_cnt else '') +
             '_gms_' + str(args.games_per_epoch)
             )
    data_folder = os.path.join(DIR, 'data', 'temp', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)
    backup_dir = os.path.join(DIR, 'data', 'save', 'backups', ident)
    retrial_fn = lambda val: args.loss_retrials if val == 0 else (args.tie_retrials if val == .5 else 0)


    def env_constructor(train_infos):
        return MyQuaticusEnv(save_video=args.save_video is not None,
                             render_mode=RENDER_MODE,
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
    create_attack = lambda i, env: (policies['att'][i](agent_id=0,
                                                       mode=modes[i%len(modes)],
                                                       using_pyquaticus=True,
                                                       ), non_train_dict.copy()
                                    )
    create_defend = lambda i, env: (policies['def'][i](agent_id=0,
                                                       team='red',
                                                       mode=modes[i%len(modes)],
                                                       flag_keepout=config_dict['flag_keepout'],
                                                       catch_radius=config_dict["catch_radius"],
                                                       using_pyquaticus=True,
                                                       ), non_train_dict.copy()
                                    )
    create_rand = lambda i, env: (RandPolicy(env.action_space), non_train_dict.copy())

    non_learning_sizes = [3,
                          3,
                          rand_cnt,
                          ]
    non_lerning_construct = [create_attack,
                             create_defend,
                             create_rand,
                             ]

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
                num_agents=sum(non_learning_sizes),
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
            track_age=True,
        ),
        # todo: args prob, but currently doing the capacity/8
        weight_decay_half_life=args.capacity/8,
        optimizer_kwargs=optim_kwargs,
    )
    trainer = ComparisionExperiment(population_sizes=non_learning_sizes,
                                    team_trainer=team_trainer,
                                    outcome_fn_gen=CTFOutcome,
                                    env_constructor=env_constructor,
                                    worker_constructors=non_lerning_construct,
                                    storage_dir=data_folder,
                                    processes=proc,
                                    team_sizes=(team_size, team_size),
                                    MCAA_mode=False,
                                    games_per_epoch=args.games_per_epoch,
                                    uniform_random_cap_select=False,
                                    # member_to_population=lambda team_idx, member_idx: {team_idx},
                                    team_member_elo_update=1*np.log(10)/400,
                                    probability_weighting=True,
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
        init_dist = 0.  # this will later become a numpy vector
        team_dist = dict()
        print('sampling distributions for plotting')
        for _ in range(args.dist_sample):
            indices, dist, og_dist = trainer.team_trainer.get_member_distribution(init_team=test_team,
                                                                                  indices=None,
                                                                                  obs_preembed=None,
                                                                                  obs_mask=None,
                                                                                  noise_model=None,
                                                                                  valid_members=None,
                                                                                  )
            # in the first iteration, this sets init_dist to a numpy vector
            # 0+np.array evaluates to np.array
            init_dist += torch.mean(dist, dim=0).detach().numpy()/args.dist_sample

            sample_team_dist = team_trainer.get_total_distribution(T=team_size, N=1)
            for team in sample_team_dist:
                if team not in team_dist:
                    team_dist[team] = 0
                team_dist[team] += sample_team_dist[team]/args.dist_sample

        plotting['init_dists'].append(init_dist)
        team_dist_non_ordered = dict()
        for item in team_dist:
            key = tuple(sorted(item))
            if key not in team_dist_non_ordered:
                team_dist_non_ordered[key] = 0
            team_dist_non_ordered[key] += team_dist[item]
        plotting['team_dists'].append(team_dist)
        plotting['team_dists_non_ordered'].append(team_dist_non_ordered)
        plotting['epochs'].append(trainer.epochs)
        print('done sampling')


    if args.plot:
        from experiments.pyquaticus_utils.dist_plot import plot_dist_evolution, order_compensate_dist

        labels = (['att easy', 'att mid', 'att hard'] +
                  ['def easy', 'def mid', 'def hard'] +
                  ['random'])
        print('plotting and closing')
        # cut off at args.epochs
        plotting['epochs'] = [epoch for epoch in plotting['epochs'] if epoch <= args.epochs]
        last_index = len(plotting['epochs']) - 1
        smoothing = args.smoothing

        init_dists = [
            np.array([t for t in dist[:6]] + [np.sum(dist[6:])])
            for dist in plotting['init_dists']
        ]
        color = ['red']*3 + ['blue']*3 + ['black']
        alpha = [.25, .5, 1] + [.25, .5, 1] + [1]
        label = labels

        perm = sorted(range(7), key=lambda i: init_dists[last_index][i], reverse=True)

        plot_dist_evolution(plot_dist=[dist[perm] for dist in init_dists],
                            x=plotting['epochs'],
                            save_dir=os.path.join(save_dir, 'initial_plot.png'),
                            title='Initial Distributions',
                            label=[labels[i] for i in perm],
                            alpha=[alpha[i] for i in perm],
                            color=[color[i] for i in perm],
                            legend_position=(-.5, .5),
                            fontsize=17,
                            smoothing=smoothing,
                            )
        for compensate in False, True:
            if compensate:
                s = 'compensated_'
                team_dists_non_ordered = copy.deepcopy(plotting['team_dists_non_ordered'])
                # we must compensate all non-orderd dicts
                team_dists_non_ordered = [order_compensate_dist(dist) for dist in team_dists_non_ordered]
            else:
                s = ''
                team_dists_non_ordered = plotting['team_dists_non_ordered']
            print(s + 'final occurence probs')

            possible_teams = sorted(team_dists_non_ordered[last_index].keys(),
                                    key=lambda k: team_dists_non_ordered[last_index][k],
                                    reverse=True,
                                    )
            occurence_probs = torch.tensor([team_dists_non_ordered[last_index][team]
                                            for team in possible_teams])
            guesstimated_elos = torch.log(occurence_probs)
            guesstimated_elos = guesstimated_elos - torch.mean(guesstimated_elos)

            elo_conversion = 400/np.log(10)
            scaled = guesstimated_elos*elo_conversion + 1000
            for i, (team, prob, elo) in enumerate(zip(possible_teams, occurence_probs, scaled)):
                print(i + 1, ':', team, ':', round(prob.item(), 2), ':', elo.item())

            all_team_dist = []
            for team_dist in team_dists_non_ordered:
                all_team_dist.append(np.array([team_dist[team]
                                               for team in possible_teams]))

            extra_text = 'KEY:\n' + '\n'.join([str(i) + ': ' + lab
                                               for i, lab in enumerate(labels[:6])])

            plot_dist_evolution(plot_dist=all_team_dist,
                                x=plotting['epochs'],
                                save_dir=os.path.join(save_dir, s + 'total_plot.png'),
                                title="Total Distribution",
                                info=extra_text + ('\n6+: random' if rand_cnt > 0 else ''),
                                legend_position=(-.3, .5 + .4/2),  # info takes up about .4
                                label=possible_teams,
                                smoothing=smoothing,
                                )

            extra_text = 'KEY:\n' + '\n'.join([str(i) + ': ' + lab
                                               for i, lab in enumerate(labels[:6])])
            cutoff = args.cutoff
            info_y = .3 + .05*cutoff
            plot_dist_evolution(plot_dist=all_team_dist,
                                x=plotting['epochs'],
                                mapping=lambda dist: np.array([t for t in dist[:cutoff]] + [np.sum(dist[cutoff:])]),
                                save_dir=os.path.join(save_dir, s + 'first_10_total_plot.png'),
                                title="Total Distribution (Top " + str(cutoff) + ")",
                                legend_position=(-.45, .2),
                                info=extra_text + ('\n6+: random' if rand_cnt > 0 else ''),
                                info_position=(-.42, info_y),
                                label=possible_teams[:cutoff] + ['other'],
                                color=[None]*cutoff + ['black'],
                                fontsize=17,
                                smoothing=smoothing,
                                )

            if rand_cnt > 0:
                # all keys that have random agents
                random_keys = [i for i, team in enumerate(possible_teams) if any([member > 5 for member in team])]
                non_random_keys = [i for i, team in enumerate(possible_teams) if all([member <= 5 for member in team])]

                plot_dist_evolution(plot_dist=all_team_dist,
                                    x=plotting['epochs'],
                                    mapping=lambda dist: np.concatenate(
                                        (dist[non_random_keys], [np.sum(dist[random_keys])])),
                                    save_dir=os.path.join(save_dir, s + 'edited_total_plot.png'),
                                    title="Total Distribution (random combined)",
                                    info=extra_text,
                                    legend_position=(-.3, .69 - .09),
                                    label=[possible_teams[i] for i in non_random_keys] + ['random'],
                                    color=[None]*len(non_random_keys) + ['black'],
                                    smoothing=smoothing,
                                    )

        # now find distribution of underlying data
        relevant = []
        for i in range(len(team_trainer.buffer)):
            (_, _, team, _, weight), age = team_trainer.buffer[i]
            relevant.append(
                (tuple(team.cpu().detach().numpy()),
                 weight,
                 age,
                 )
            )
        dist = {}
        wdist = {}
        decaying_wdist = {}
        for team, weight, age in relevant:
            for d in dist, wdist, decaying_wdist:
                if team not in d:
                    d[team] = 0
            dist[team] += 1
            weight = weight.item()
            wdist[team] += weight
            decaying_wdist[team] += weight*torch.pow(torch.tensor(1/2), age/team_trainer.weight_decay_half_life).item()
        for d in dist, wdist, decaying_wdist:
            s = 0
            for team in d:
                s += d[team]
            for team in d:
                d[team] = d[team]/s
            keys = list(d.keys())
            keys.sort(key=lambda team: d[team], reverse=True)
        for _ in range(1):
            print()
            print('trinin')
            trainer.team_trainer.train(
                batch_size=args.batch_size,
                minibatch=args.minibatch_size,
                mask_probs=None,
                replacement_probs=(.8, .1, .1),
                mask_obs_prob=.1,
            )
            berteam_dist = team_trainer.get_total_distribution(T=2)

            keys = list(wdist.keys())
            keys.sort(key=lambda team: d[team], reverse=True)
            overall_dist_err = 0
            for key in keys:
                print(key,
                      'berteam', round(berteam_dist[key], 2),
                      '\td', round(dist[key], 2),
                      '\twd', round(wdist[key], 2),
                      '\tdwd', round(decaying_wdist[key], 2),
                      '\terr', round(berteam_dist[key] - decaying_wdist[key], 2),
                      )
                overall_dist_err += abs(berteam_dist[key] - decaying_wdist[key])
            print('overall abs error', overall_dist_err)
        trainer.clear()
        quit()


    def typer(global_idx):
        animal, _ = trainer.load_animal(trainer.index_to_pop_index(global_idx))
        if 'WrappedPolicy' in str(type(animal)):
            return animal.identity
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
        print('agent list')
        for i in range(6):
            print(str(i) + ':', typer(i))
        if rand_cnt > 0:
            if rand_cnt > 1:
                print('6-' + str(6 + rand_cnt - 1) + ':', typer(6))
            else:
                print('6:', typer(6))
        env = env_constructor(None)
        if idxs is None:
            gen_team = [t.item()
                        for t in trainer.team_trainer.create_teams(T=team_size).flatten()]
            print('generated team has elos', trainer.classic_elos[gen_team])

            print('best agent has elo', trainer.classic_elos[best],
                  'and is type', typer(best))
            print('worst agents has elo', trainer.classic_elos[worst],
                  'and is type', typer(worst))

            print('playing generated (blue, ' + str([typer(idx) for idx in gen_team])
                  + ') against double best (red, ' + str([typer(best) for idx in gen_team]) + ')')

            outcom = CTFOutcome()
            agents = []
            for team in gen_team, [best]*team_size:
                m = []
                for idx in team:
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
            A, B = [ast.literal_eval('(' + team + ')') for team in idxs.split(';')]

            print('playing', A, '(blue, ' + str([typer(idx) for idx in A])
                  + ') against', B, '(red, ' + str([typer(idx) for idx in B]) + ')')

            outcom = CTFOutcome()
            agents = []
            for team in A, B:
                m = []
                for idx in team:
                    agent = trainer.load_animal(trainer.index_to_pop_index(idx))[0]
                    m.append(agent)
                agents.append(m)

            outcom.get_outcome(
                team_choices=[[torch.tensor(idx) for idx in A], [torch.tensor(idx)]*team_size],
                agent_choices=agents,
                env=env,
                updated_train_infos=[[non_train_dict]*team_size]*2,
            )

        if args.save_video is not None:
            env.write_video(video_file=args.save_video)
    else:
        while trainer.epochs < args.epochs:
            tim = time.time()
            print('starting epoch', trainer.info['epochs'], 'at time', time.strftime('%H:%M:%S'))
            # print(trainer.team_trainer.get_total_distribution(T=team_size))
            # quit()
            if trainer.epochs == 0:
                update_plotting_variables()
            epoch_info = trainer.epoch(
                noise_model=team_trainer.create_nose_model_towards_uniform(
                    t=torch.exp(-np.log(2.)*trainer.ages/args.half_life)
                )
                ,
                save_epoch_info=False,  # TODO: probably can do this more elegantly
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
                for i in range(sum(non_learning_sizes)):
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

            if not (trainer.info['epochs'])%args.ckpt_freq:
                if not args.dont_backup and os.path.exists(save_dir):
                    print('backing up')
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)
                    shutil.copytree(save_dir, backup_dir)
                    print('done backing up')
                print('saving')
                trainer.save(save_dir)

                f = open(os.path.join(save_dir, 'plotting.pkl'), 'wb')
                pickle.dump(plotting, f)
                f.close()
                print('done saving')

            print('time', time.time() - tim)
            print()

    trainer.clear()

    shutil.rmtree(data_folder)
