import argparse


def add_pyquaticus_args(PARSER: argparse.ArgumentParser, arena_size=True):
    if arena_size:
        PARSER.add_argument('--arena-size', action='store', required=False, default='200.0,100.0',
                            help="x,y dims of arena, in format '200.0,100.0'")

    PARSER.add_argument('--max-time', type=float, required=False, default=420.,
                        help="max sim time of each episode")
    PARSER.add_argument('--sim-speedup-factor', type=int, required=False, default=40,
                        help="skips frames to speed up episodes")
    PARSER.add_argument('--unnormalize', action='store_true', required=False,
                        help="do not normalize, arg is necessary to use some pyquaticus bots")
    PARSER.add_argument('--unblock-gpu', action='store_true', required=False,
                        help="unblock using gpu ")


def add_team_args(PARSER: argparse.ArgumentParser):
    PARSER.add_argument('--team-size', type=int, required=False, default=2,
                        help="size of teams")


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


def add_coevolution_args(PARSER: argparse.ArgumentParser):
    PARSER.add_argument('--protect-new', type=int, required=False, default=500,
                        help="protect new agents for this number of breeding epochs")
    PARSER.add_argument('--mutation-prob', type=float, required=False, default=0.,
                        help="probabality of mutating agents each epoch (should probably be very small)")
    PARSER.add_argument('--clone-replacements', type=int, required=False, default=None,
                        help="number of agents to try replacing each epoch (default all)")


def add_berteam_args(PARSER: argparse.ArgumentParser):
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
    PARSER.add_argument('--minibatch-size', type=int, required=False, default=1,
                        help="minibatch size for Multi Level Marketing training (1 is equivalent to sgd)")
    PARSER.add_argument('--train-freq', type=int, required=False, default=10,
                        help="train freq for Multi Level Marketing training")

    PARSER.add_argument('--half-life', type=float, required=False, default=400.,
                        help="noise goes from uniform distribution to this wrt agents age")

    PARSER.add_argument('--loss-retrials', type=int, required=False, default=0,
                        help="number of times to retry a loss")
    PARSER.add_argument('--tie-retrials', type=int, required=False, default=0,
                        help="number of times to retry a tie")


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
                        help='which agent indexes to display, in the format "i1,i2;j1,j2" (used with --display)')
    PARSER.add_argument('--render', action='store_true', required=False,
                        help="Enable rendering")
    PARSER.add_argument('--seed', type=int, required=False, default=0,
                        help="random seed")
