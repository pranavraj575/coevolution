import argparse, ast


def add_pyquaticus_args(PARSER: argparse.ArgumentParser, arena_size=True, flag_keepout=True):
    if arena_size:
        PARSER.add_argument('--arena-size', action='store', required=False, default='200.0,100.0',
                            help='x,y dims of arena, in format "(200.0,100.0)"')
    if flag_keepout:
        PARSER.add_argument('--flag-keepout', type=float, required=False, default=5.,
                            help="radius that defending agents must avoid flag by")
    PARSER.add_argument('--max-time', type=float, required=False, default=420.,
                        help="max sim time of each episode")
    PARSER.add_argument('--sim-speedup-factor', type=int, required=False, default=10,
                        help="skips frames to speed up episodes")
    PARSER.add_argument('--unnormalize', action='store_true', required=False,
                        help="do not normalize, arg is necessary to use some pyquaticus bots")
    PARSER.add_argument('--unblock-gpu', action='store_true', required=False,
                        help="unblock using gpu ")


def add_team_args(PARSER: argparse.ArgumentParser):
    PARSER.add_argument('--team-size', type=int, required=False, default=2,
                        help="size of teams")


def pyquaticus_string(args):
    thing = ''
    if 'team_size' in dir(args):
        thing += '_tm_sz_' + str(args.team_size)
    if 'arena_size' in dir(args):
        arena_size = ast.literal_eval('(' + args.arena_size + ')')
        arena_size = tuple(float(t) for t in arena_size)
        thing += '_arena_' + str('__'.join([str(t).replace('.', '_') for t in arena_size]))
    if 'flag_keepout' in dir(args):
        if args.flag_keepout != 5.:
            print("WARNING FLAG KEEPOUT IS DIFFERENT MAKE SURE YOU CHANGED PYQUATICUS")
            thing += '_kpout_' + str(float(args.flag_keepout)).replace('.', '_')
    if args.unnormalize:
        thing += '_no_norm_obs'
    return thing


def update_config_dict(config_dict, args):
    config_dict["max_screen_size"] = (float('inf'), float('inf'))
    if 'arena_size' in dir(args):
        arena_size = ast.literal_eval('(' + args.arena_size + ')')
        arena_size = tuple(float(t) for t in arena_size)
        config_dict["world_size"] = arena_size
    if 'flag_keepout' in dir(args):
        config_dict['flag_keepout'] = args.flag_keepout
    normalize = not args.unnormalize

    config_dict["sim_speedup_factor"] = args.sim_speedup_factor
    config_dict["max_time"] = args.max_time
    config_dict['normalize'] = normalize


def add_learning_agent_args(PARSER: argparse.ArgumentParser, split_learners=False):
    PARSER.add_argument('--ppo-agents', type=int, required=False, default=0,
                        help="number of ppo agents to use")
    PARSER.add_argument('--dqn-agents', type=int, required=False, default=0,
                        help="number of dqn agents to use")
    PARSER.add_argument('--replay-buffer-capacity', type=int, required=False, default=10000,
                        help="replay buffer capacity")
    PARSER.add_argument('--net-arch', action='store', required=False, default='64,64',
                        help="hidden layers of network, should be readable as a list or tuple")
    if split_learners:
        PARSER.add_argument('--split-learners', action='store_true', required=False,
                            help="learning agents types each go in their own population to avoid interspecies replacement")


def learning_agents_string(args):
    net_arch = tuple(ast.literal_eval('(' + args.net_arch + ')'))
    buffer_cap = args.replay_buffer_capacity

    thing = ((('_arch_' + '_'.join([str(s) for s in net_arch])) if args.ppo_agents + args.dqn_agents else '') +
             (('_ppo_' + str(args.ppo_agents)) if args.ppo_agents else '') +
             (('_dqn_' + str(args.dqn_agents)) if args.dqn_agents else '') +
             '_' +
             (('_rb_cap_' + str(buffer_cap)) if args.dqn_agents else ''))
    if 'split_learners' in dir(args) and args.split_learners and args.ppo_agents and args.dqn_agents:
        thing += '_split_'
    return thing


def add_coevolution_args(PARSER: argparse.ArgumentParser, clone_default=None):
    PARSER.add_argument('--protect-new', type=int, required=False, default=500,
                        help="protect new agents for this number of breeding epochs")
    PARSER.add_argument('--mutation-prob', type=float, required=False, default=0.,
                        help="probabality of mutating agents each epoch (should probably be very small)")
    PARSER.add_argument('--clone-replacements', type=int, required=False, default=clone_default,
                        help="number of agents to try replacing each epoch (default all)")


def add_berteam_args(PARSER: argparse.ArgumentParser):
    PARSER.add_argument('--embedding-dim', type=int, required=False, default=128,
                        help="size of transformer embedding layer")

    PARSER.add_argument('--heads', type=int, required=False, default=4,
                        help="transformer number of attention heads to use")
    PARSER.add_argument('--encoders', type=int, required=False, default=3,
                        help="transformer number of decoder layers")
    PARSER.add_argument('--decoders', type=int, required=False, default=3,
                        help="transformer number of decoder layers")
    PARSER.add_argument('--dropout', type=float, required=False, default=.1,
                        help="transformer dropout")

    PARSER.add_argument('--lstm-layers', type=int, required=False, default=2,
                        help="LSTM number of layers")
    PARSER.add_argument('--lstm-dropout', type=int, required=False, default=None,
                        help="LSTM dropout (defaults to transformer dropout)")

    PARSER.add_argument('--capacity', type=int, required=False, default=5000,
                        help="capacity of game replay buffer")

    PARSER.add_argument('--batch-size', type=int, required=False, default=1024,
                        help="batch size for Multi Level Marketing training")
    PARSER.add_argument('--minibatch-size', type=int, required=False, default=256,
                        help="minibatch size for Multi Level Marketing training (1 is equivalent to sgd)")
    PARSER.add_argument('--train-freq', type=int, required=False, default=10,
                        help="train freq for Multi Level Marketing training")

    PARSER.add_argument('--half-life', type=float, required=False, default=400.,
                        help="noise goes from uniform distribution to this wrt agents age")

    PARSER.add_argument('--loss-retrials', type=int, required=False, default=0,
                        help="number of times to retry a loss")
    PARSER.add_argument('--tie-retrials', type=int, required=False, default=0,
                        help="number of times to retry a tie")


def berteam_string(args):
    return ('_embed_dim_' + str(args.embedding_dim) +
            '_trans_' +
            (
                    '_head_' + str(args.heads) +
                    '_enc_' + str(args.encoders) +
                    '_dec_' + str(args.decoders) +
                    '_drop_' + str(args.dropout).replace('.', '_')
            ) +
            '_inp_emb_' +
            (
                    '_lyrs_' + str(args.lstm_layers) +
                    ('_drop_' + args.lstm_dropout if args.lstm_dropout is not None else '')
            ) +
            (
                ('_retr_' +
                 ('_l_' + str(args.loss_retrials) if args.loss_retrials else '') +
                 ('_t_' + str(args.tie_retrials) if args.tie_retrials else '')
                 )
                if args.loss_retrials or args.tie_retrials else ''
            ) +
            '_train_frq_' + str(args.train_freq) +
            '_btch_' + str(args.batch_size) +
            '_minibtch_' + str(args.minibatch_size) +
            '_half_life_' + str(float(args.half_life)).replace('.', '_'))


def add_experiment_args(PARSER: argparse.ArgumentParser, ident):
    PARSER.add_argument('--epochs', type=int, required=False, default=5000,
                        help="epochs to train for")
    PARSER.add_argument('--processes', type=int, required=False, default=0,
                        help="number of processes to use")

    PARSER.add_argument('--ident', action='store', required=False, default=ident,
                        help='identification to add to folder')
    PARSER.add_argument('--ckpt_freq', type=int, required=False, default=25,
                        help="checkpoint freq")
    PARSER.add_argument('--reset', action='store_true', required=False,
                        help="do not load from save")

    PARSER.add_argument('--display', action='store_true', required=False,
                        help="skip training and display saved model")
    PARSER.add_argument('--idxs-to-display', action='store', required=False, default=None,
                        help='which agent indexes to display, in the format "(i1,i2);(j1,j2)" (used with --display)')
    PARSER.add_argument('--render', action='store_true', required=False,
                        help="Enable rendering")
    PARSER.add_argument('--seed', type=int, required=False, default=0,
                        help="random seed")
