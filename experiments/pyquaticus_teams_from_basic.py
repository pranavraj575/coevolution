import argparse

if __name__ == '__main__':
    import torch, random
    import numpy as np

    torch.random.manual_seed(0)
    np.random.seed()
    random.seed()

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--team-size', type=int, required=False, default=2,
                        help="size of teams")

    PARSER.add_argument('--embedding-dim', type=int, required=False, default=128,
                        help="size of transformer embedding layer")

    PARSER.add_argument('--heads', type=int, required=False, default=8,
                        help="transformer number of attention heads to use")
    PARSER.add_argument('--encoders', type=int, required=False, default=8,
                        help="transformer number of decoder layers")
    PARSER.add_argument('--decoders', type=int, required=False, default=8,
                        help="transformer number of decoder layers")
    PARSER.add_argument('--dropout', type=float, required=False, default=.1,
                        help="transformer dropout")

    PARSER.add_argument('--lstm-layers', type=int, required=False, default=4,
                        help="LSTM number of layers")
    PARSER.add_argument('--lstm-dropout', type=int, required=False, default=None,
                        help="LSTM dropout (defaults to transformer dropout)")

    PARSER.add_argument('--capacity', type=int, required=False, default=1e4,
                        help="capacity of game replay buffer")

    PARSER.add_argument('--batch-size', type=int, required=False, default=128,
                        help="batch size for Multi Level Marketing training")
    PARSER.add_argument('--train-freq', type=int, required=False, default=10,
                        help="train freq for Multi Level Marketing training")

    PARSER.add_argument('--half-life', type=float, required=False, default=400,
                        help="noise goes from uniform distribution to this wrt agents age")

    PARSER.add_argument('--loss-retrials', type=int, required=False, default=0,
                        help="number of times to retry a loss")
    PARSER.add_argument('--tie-retrials', type=int, required=False, default=0,
                        help="number of times to retry a tie")

    PARSER.add_argument('--max-time', type=float, required=False, default=420.,
                        help="max sim time of each episode")
    PARSER.add_argument('--sim-speedup-factor', type=int, required=False, default=40,
                        help="skips frames to speed up episodes")
    PARSER.add_argument('--unnormalize', action='store_true', required=False,
                        help="do not normalize, arg is necessary to use some pyquaticus bots")

    PARSER.add_argument('--rand-count', type=int, required=False, default=0,
                        help="number of random agents to add into population to try confusing team selector")

    PARSER.add_argument('--epochs', type=int, required=False, default=5000,
                        help="epochs to train for")
    PARSER.add_argument('--processes', type=int, required=False, default=0,
                        help="number of processes to use")

    PARSER.add_argument('--ident', action='store', required=False, default='pyquaticus_team_MLM',
                        help='identification to add to folder')
    PARSER.add_argument('--reset', action='store_true', required=False,
                        help="do not load from save")
    PARSER.add_argument('--ckpt_freq', type=int, required=False, default=25,
                        help="checkpoint freq")

    PARSER.add_argument('--render', action='store_true', required=False,
                        help="Enable rendering")
    PARSER.add_argument('--display', action='store_true', required=False,
                        help="skip training and display saved model")
    PARSER.add_argument('--idxs-to-display', action='store', required=False, default=None,
                        help='which agent indexes to display, in the format "p1,p2;p1,p2" (used with --display)')
    args = PARSER.parse_args()

    import torch, os, sys, ast, time, random

    from repos.pyquaticus.pyquaticus.config import config_dict_std
    from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
    from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker

    from experiments.pyquaticus_utils.reward_fns import custom_rew, custom_rew2, RandPolicy
    from experiments.pyquaticus_utils.wrappers import MyQuaticusEnv, policy_wrapper
    from experiments.pyquaticus_utils.outcomes import CTFOutcome

    from networks.positional_encoder import IdentityEncoding, ClassicPositionalEncoding, PositionalAppender
    from networks.input_embedding import LSTEmbedding

    from src.language_replay_buffer import BinnedReplayBufferDiskStorage
    from src.team_trainer import MLMTeamTrainer, BERTeam, TeamBuilder

    from src.coevolver import PettingZooCaptianCoevolution
    from src.utils.dict_keys import *

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    team_size = args.team_size

    reward_config = {i: custom_rew2 for i in range(2*team_size)}  # Example Reward Config

    config_dict = config_dict_std
    config_dict["max_screen_size"] = (float('inf'), float('inf'))

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

    config_dict["sim_speedup_factor"] = args.sim_speedup_factor
    config_dict["max_time"] = args.max_time
    RENDER_MODE = 'human' if args.render or args.display else None

    normalize = not args.unnormalize
    config_dict['normalize'] = normalize

    lstm_dropout = args.lstm_dropout
    if lstm_dropout is None:
        lstm_dropout = args.dropout
    rand_cnt = args.rand_count
    ident = (args.ident +
             '_team_size_' + str(team_size) +
             '_embedding_dim_' + str(args.embedding_dim) +
             '_transformer_' +
             (
                     '_heads_' + str(args.heads) +
                     '_encoders_' + str(args.encoders) +
                     '_decoders_' + str(args.decoders) +
                     '_dropout_' + str(args.dropout).replace('.', '_')
             ) +
             '_input_embedder_' +
             (
                     '_layers_' + str(args.lstm_layers) +
                     ('_dropout_' + lstm_dropout if args.lstm_dropout is not None else '')
             ) +
             (('_random_agents_' + str(rand_cnt)) if rand_cnt else '') +
             '_train_freq_' + str(args.train_freq) +
             '_loss_retrials_' + str(args.loss_retrials) +
             '_tie_retrials_' + str(args.tie_retrials) +
             ('_dont_normalize_obs' if not normalize else '')
             )
    data_folder = os.path.join(DIR, 'data', 'temp', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)
    retrial_fn = lambda val: args.loss_retrials if val == 0 else (args.tie_retrials if val == .5 else 0)


    def env_constructor(train_infos):
        return MyQuaticusEnv(render_mode=RENDER_MODE,
                             reward_config=reward_config,
                             team_size=team_size,
                             config_dict=config_dict,
                             )


    non_train_dict = {DICT_TRAIN: False,
                      DICT_CLONABLE: False,
                      DICT_CLONE_REPLACABLE: False,
                      DICT_MUTATION_REPLACABLE: False,
                      DICT_IS_WORKER: False,
                      }
    create_rand = lambda i, env: (RandPolicy(env.action_space), non_train_dict.copy())
    create_defend = lambda i, env: (policies['def'][i](agent_id=0,
                                                       team='red',
                                                       mode=modes[i%len(modes)],
                                                       flag_keepout=config_dict['flag_keepout'],
                                                       catch_radius=config_dict["catch_radius"],
                                                       using_pyquaticus=True,
                                                       ), non_train_dict.copy()
                                    )
    create_attack = lambda i, env: (policies['att'][i](agent_id=0,
                                                       mode=modes[i%len(modes)],
                                                       using_pyquaticus=True,
                                                       ), non_train_dict.copy()
                                    )

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
        )
    )
    trainer = PettingZooCaptianCoevolution(population_sizes=non_learning_sizes,
                                           team_trainer=team_trainer,
                                           outcome_fn_gen=CTFOutcome,
                                           env_constructor=env_constructor,
                                           worker_constructors=non_lerning_construct,
                                           storage_dir=data_folder,
                                           processes=proc,
                                           team_sizes=(team_size, team_size),
                                           depth_to_retry_result=retrial_fn,
                                           # member_to_population=lambda team_idx, member_idx: {team_idx},

                                           team_member_elo_update=1*np.log(10)/400,
                                           )

    if not args.reset and os.path.exists(save_dir):
        print('loading from', save_dir)
        trainer.load(save_dir=save_dir)
    print('seting save directory as', save_dir)


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
        if idxs is None:
            classic_elos = trainer.classic_elos.numpy()
            idents_and_elos = []
            print('all elos by index')
            for i, (identity, elo) in enumerate(idents_and_elos):
                print(i, ' (', identity, '): ', elo, sep='')

        else:
            team_1, team_2 = [ast.literal_eval(team) for team in idxs.split(';')]

            print('playing', team_1, '(blue) against', team_2, '(red)')
    else:
        while trainer.epochs < args.epochs:
            tim = time.time()
            print('starting epoch', trainer.info['epochs'], 'at time', time.strftime('%H:%M:%S'))

            epoch_info = trainer.epoch(
                noise_model=team_trainer.create_nose_model_towards_uniform(
                    t=torch.exp(-np.log(2.)*trainer.ages/args.half_life)
                )
            )
            if not trainer.epochs%args.train_freq:
                print('training step started')
                trainer.team_trainer.training_step(
                    batch_size=args.batch_size,
                )
                print('training step finished')
            if True:
                classic_elos = trainer.classic_elos.numpy()
                print('all elos')
                print(classic_elos)
            if not (trainer.info['epochs'])%args.ckpt_freq:
                print('saving')
                trainer.save(save_dir)
                print('done saving')
            print('time', time.time() - tim)
            print()

    trainer.clear()
    import shutil

    shutil.rmtree(data_folder)
