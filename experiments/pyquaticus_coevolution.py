import argparse, os, sys, ast
import torch.random
from unstable_baselines3 import WorkerPPO, WorkerDQN
from unstable_baselines3.common.auto_multi_alg import AutoMultiAgentAlgorithm
from unstable_baselines3.dqn import MlpPolicy as DQNMlp
from unstable_baselines3.ppo import MlpPolicy as PPOMlp

from repos.pyquaticus.pyquaticus import pyquaticus_v0
from repos.pyquaticus.pyquaticus.config import config_dict_std
from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
from experiments.pyquaticus_utils.utils import policy_wrapper

from src.coevolver import PettingZooCaptianCoevolution
from src.game_outcome import PettingZooOutcomeFn
from src.utils.dict_keys import *


def custom_rew(self, params, prev_params):
    if params["agent_tagged"][params["agent_id"]]:
        return 0
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        return 10
    if params['has_flag']:
        diff = prev_params['team_flag_home'] - params['team_flag_home']
        diff = diff/5
        return .1 + min(.9, .9*diff)
    if params['opponent_flag_distance'] > 0:
        diff = prev_params['opponent_flag_distance'] - params['opponent_flag_distance']
        diff = diff/5
        return min(.1, .1*diff)
    return 0


def custom_rew2(self, params, prev_params):
    reward = 0
    
    flag_capture_rew=1.  
    flag_pickup_rew=1.
    tag_reward=.05
    oob_penalty=1.
    
    # Penalize player for opponent grabbing team flag
    if params["opponent_flag_pickup"] and not prev_params["opponent_flag_pickup"]:
        reward += -flag_pickup_rew
    # Reward player for grabbing opponents flag
    if params["has_flag"] and not prev_params['has_flag']:
        reward += flag_pickup_rew
    
    # penalize player for dropping flag
    if not params["team_flag_capture"] and (prev_params["has_flag"] and not params['has_flag']):
        reward += -flag_pickup_rew
                         
    # Penalize player for opponent successfully capturing team flag
    if params["opponent_flag_capture"] and not prev_params["opponent_flag_capture"]:
        reward += -flag_capture_rew
    # Reward player for capturing opponents flag
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        reward += flag_capture_rew

    # Check to see if agent was tagged
    if params["agent_tagged"][params["agent_id"]] and not prev_params["agent_tagged"][params["agent_id"]]:
        reward+= -tag_reward
    # Check to see if agent tagged an opponent
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        reward += tag_reward
        if prev_params["opponent_" + str(tagged_opponent) + "_has_flag"]:
            reward += flag_pickup_rew
    
    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward += -oob_penalty
    return reward


class CTFOutcome(PettingZooOutcomeFn):
    def _get_outcome_from_agents(self, agent_choices, index_choices, updated_train_infos, env):
        agent_choices = agent_choices[0] + agent_choices[1]
        updated_train_infos = updated_train_infos[0] + updated_train_infos[1]

        # env is set up so the first k agents are team blue and the last k agents are team red
        alg = AutoMultiAgentAlgorithm(env=env,
                                      workers={i: agent_choices[i] for i in range(len(agent_choices))},
                                      worker_infos={i: updated_train_infos[i] for i in range(len(agent_choices))},
                                      )
        alg.learn(total_timesteps=10000,
                  number_of_eps=1,
                  )
        score = (env.unwrapped.game_score['blue_captures'], env.unwrapped.game_score['red_captures'])

        if score[0] == score[1]:
            return [
                (.5, []),
                (.5, []),
            ]
        if score[0] > score[1]:
            return [
                (1, []),
                (0, []),
            ]
        if score[0] < score[1]:
            return [
                (0, []),
                (1, []),
            ]


class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs, *args, **kwargs):
        return self.action_space.sample()

DefendPolicy=policy_wrapper(BaseDefender)

config_dict = config_dict_std
config_dict["max_screen_size"] = (float('inf'), float('inf'))
# config_dict['tag_on_wall_collision']=True
reward_config = {0: custom_rew2, 1: custom_rew2, 5: None}  # Example Reward Config

if __name__ == '__main__':
    import time
    import numpy as np
    import random

    torch.random.manual_seed(0)
    np.random.seed()
    random.seed()

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--render', action='store_true', required=False,
                        help="Enable rendering")
    PARSER.add_argument('--reset', action='store_true', required=False,
                        help="do not load from save")

    PARSER.add_argument('--unnormalize', action='store_true', required=False,
                        help="do not normalize, arg is necessary to use some pyquaticus bots")

    PARSER.add_argument('--ckpt_freq', type=int, required=False, default=25,
                        help="checkpoint freq")

    PARSER.add_argument('--epochs', type=int, required=False, default=5000,
                        help="epochs to train for")

    PARSER.add_argument('--rand-agents', type=int, required=False, default=25,
                        help="number of random agents to use")
    PARSER.add_argument('--ppo-agents', type=int, required=False, default=25,
                        help="number of ppo agents to use")
    PARSER.add_argument('--dqn-agents', type=int, required=False, default=25,
                        help="number of dqn agents to use")
    PARSER.add_argument('--replay-buffer-capacity', type=int, required=False, default=10000,
                        help="replay buffer capacity")

    PARSER.add_argument('--net-arch', action='store', required=False, default='128,64',
                        help="hidden layers of network, should be readable as a list or tuple")

    PARSER.add_argument('--split-learners', action='store_true', required=False,
                        help="learning agents types each go in their own population to avoid interspecies replacement")

    PARSER.add_argument('--protect-new', type=int, required=False, default=300,
                        help="protect new agents for this number of breeding epochs")

    PARSER.add_argument('--max-time', type=float, required=False, default=420.,
                        help="max sim time of each episode")
    PARSER.add_argument('--sim-speedup-factor', type=int, required=False, default=40,
                        help="skips frames to speed up episodes")

    PARSER.add_argument('--processes', type=int, required=False, default=0,
                        help="number of processes to use")
    PARSER.add_argument('--ident', action='store', required=False, default='pyquaticus_coevolution',
                        help='identification to add to folder')

    PARSER.add_argument('--display', action='store_true', required=False,
                        help="skip training and display saved model")
    args = PARSER.parse_args()
    config_dict["sim_speedup_factor"] = args.sim_speedup_factor
    config_dict["max_time"] = args.max_time
    RENDER_MODE = 'human' if args.render or args.display else None
    rand_cnt = args.rand_agents
    ppo_cnt = args.ppo_agents
    dqn_cnt = args.dqn_agents
    buffer_cap = args.replay_buffer_capacity
    net_arch = tuple(ast.literal_eval('(' + args.net_arch + ')'))
    normalize = not args.unnormalize
    config_dict['normalize'] = normalize

    ident = (args.ident +
             '_rand_agents_' + str(rand_cnt) +
             '_ppo_agents_' + str(ppo_cnt) +
             '_dqn_agents_' + str(dqn_cnt) +
             '_replay_buffer_capacity_' + str(buffer_cap) +
             '_split_learners_' + str(args.split_learners) +
             '_protect_new_' + str(args.protect_new) +
             '_net_arch_' + '_'.join([str(s) for s in net_arch]) +
             '_normalize_obs_' + str(normalize)
             )

    data_folder = os.path.join(DIR, 'data', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)


    def env_constructor(train_infos):
        return pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE,
                                           reward_config=reward_config,
                                           team_size=1,
                                           config_dict=config_dict,
                                           )


    create_rand = lambda i, env: (RandPolicy(env.action_space), {DICT_TRAIN: False,
                                                                 DICT_CLONABLE: False,
                                                                 DICT_CLONE_REPLACABLE: False,
                                                                 DICT_MUTATION_REPLACABLE: False,
                                                                 DICT_IS_WORKER: False,
                                                                 })

    info_dict = {DICT_TRAIN: True,
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
                                           ), info_dict.copy()
                                 )
    dqnkwargs = {
        'buffer_size': buffer_cap,
    }
    create_dqn = lambda i, env: (WorkerDQN(policy=DQNMlp,
                                           env=env,
                                           **dqnkwargs
                                           ), info_dict.copy()
                                 )

    if args.split_learners:
        pop_sizes = [rand_cnt,
                     ppo_cnt,
                     dqn_cnt,
                     ]

        worker_constructors = [create_rand,
                               create_ppo,
                               create_dqn]
    else:
        pop_sizes = [rand_cnt,
                     ppo_cnt + dqn_cnt,
                     ]


        def create_learner(i, env):
            if i < ppo_cnt:
                return create_ppo(i, env)
            else:
                return create_dqn(i - ppo_cnt, env)


        worker_constructors = [create_rand,
                               create_learner,
                               ]

    max_cores = len(os.sched_getaffinity(0))
    if args.display:
        proc = 0
    else:
        proc = args.processes
    trainer = PettingZooCaptianCoevolution(population_sizes=pop_sizes,
                                           outcome_fn_gen=CTFOutcome,
                                           env_constructor=env_constructor,
                                           worker_constructors=worker_constructors,
                                           zoo_dir=os.path.join(data_folder, 'zoo'),
                                           protect_new=args.protect_new,
                                           processes=proc,
                                           # member_to_population=lambda team_idx, member_idx: {team_idx},
                                           max_steps_per_ep=(1 + config_dict['render_fps']*config_dict['max_time']/
                                                             config_dict['sim_speedup_factor'])
                                           )

    if not args.reset and os.path.exists(save_dir):
        print('loading from', save_dir)
        trainer.load(save_dir=save_dir)


    def typer(global_idx):
        animal, _ = trainer.load_animal(trainer.index_to_pop_index(global_idx))
        if isinstance(animal, WorkerPPO):
            return 'ppo'
        elif isinstance(animal, WorkerDQN):
            return 'dqn'
        else:
            return 'random'


    if args.display:
        elos = trainer.classic_elos.numpy().copy()
        worst = np.argmin(elos)
        elos[worst] = -np.inf
        best = np.argmax(elos)
        elos[best] = -np.inf
        second_best = np.argmax(elos)
        print()
        print('best agent has elo', trainer.classic_elos[best], 'and is type', typer(best))
        print('second best agent has elo', trainer.classic_elos[second_best], 'and is type', typer(second_best))
        print('worst agent has elo', trainer.classic_elos[worst], 'and is type', typer(worst))

        print('playing worst (blue, ' + typer(worst) + ') against best (red, ' + typer(best) + ')')

        ep = trainer.pre_episode_generation(captian_choices=(worst, best), unique=(True, True))
        trainer.epoch(rechoose=False,
                      save_epoch_info=False,
                      pre_ep_dicts=[ep],
                      )
        print('playing second best (blue, ' + typer(second_best) + ') against best (red, ' + typer(best) + ')')
        ep = trainer.pre_episode_generation(captian_choices=(second_best, best), unique=(True, True))
        trainer.epoch(rechoose=False,
                      save_epoch_info=False,
                      pre_ep_dicts=[ep],
                      )
    else:
        while trainer.epochs < args.epochs:
            tim = time.time()
            print('starting epoch', trainer.info['epochs'], 'at time', time.strftime('%H:%M:%S'))
            trainer.epoch()
            classic_elos = trainer.classic_elos.numpy()
            if True:
                print('all elos')
                ppos = []
                dqns = []
                rands = []
                for i in range(sum(pop_sizes)):
                    identity = typer(i)
                    if identity == 'dqn':
                        dqns.append(i)
                    elif identity == 'ppo':
                        ppos.append(i)
                    else:
                        rands.append(i)
                print('\telo of random agents:', classic_elos[rands])
                print('\telo of', len(ppos), 'ppo agents:', classic_elos[ppos])
                print('\telo of', len(dqns), 'dqn agents:', classic_elos[dqns])

                print('avg elos')
                print('\trandom:', np.mean(classic_elos[rands]))
                print('\tppo:', np.mean(classic_elos[ppos]))
                print('\tdqn:', np.mean(classic_elos[dqns]))

            if not (trainer.info['epochs'])%args.ckpt_freq:
                print('saving')
                trainer.save(save_dir)
                print('done saving')
            print('time', time.time() - tim)
            print()

    trainer.clear()
