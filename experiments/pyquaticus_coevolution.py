import argparse, os, sys, ast
import torch.random
from unstable_baselines3 import WorkerPPO, WorkerDQN
from unstable_baselines3.dqn import MlpPolicy as DQNMlp
from unstable_baselines3.ppo import MlpPolicy as PPOMlp

from repos.pyquaticus.pyquaticus import pyquaticus_v0
from repos.pyquaticus.pyquaticus.config import config_dict_std
from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker
from repos.pyquaticus.pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent as BaseBalanced
from experiments.pyquaticus_utils.utils import policy_wrapper, custom_rew, custom_rew2, RandPolicy, CTFOutcome
from repos.pyquaticus.pyquaticus.structs import Team

from src.coevolver import PettingZooCaptianCoevolution
from src.utils.dict_keys import *

config_dict = config_dict_std
config_dict["max_screen_size"] = (float('inf'), float('inf'))
# reset later
config_dict["sim_speedup_factor"] = 40
config_dict["max_time"] = 420.
# config_dict['tag_on_wall_collision']=True
reward_config = {0: custom_rew2, 1: custom_rew2, 5: None}  # Example Reward Config

test_env = pyquaticus_v0.PyQuaticusEnv(render_mode=None,
                                       team_size=1,
                                       config_dict=config_dict,
                                       )
obs_normalizer = test_env.agent_obs_normalizer
DefendPolicy = policy_wrapper(BaseDefender, agent_obs_normalizer=obs_normalizer, identity='defender')
AttackPolicy = policy_wrapper(BaseAttacker, agent_obs_normalizer=obs_normalizer, identity='attacker')
BalancedPolicy = policy_wrapper(BaseBalanced, agent_obs_normalizer=obs_normalizer, identity='balanced')

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

    PARSER.add_argument('--rand-agents', type=int, required=False, default=0,
                        help="number of random agents to use")

    PARSER.add_argument('--defend-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    PARSER.add_argument('--attack-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    PARSER.add_argument('--combined-agents', type=int, required=False, default=0,
                        help="number of random agents to use")

    PARSER.add_argument('--ppo-agents', type=int, required=False, default=0,
                        help="number of ppo agents to use")
    PARSER.add_argument('--dqn-agents', type=int, required=False, default=0,
                        help="number of dqn agents to use")
    PARSER.add_argument('--replay-buffer-capacity', type=int, required=False, default=10000,
                        help="replay buffer capacity")

    PARSER.add_argument('--net-arch', action='store', required=False, default='64,64',
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

    defend_cnt = args.defend_agents
    attack_cnt = args.attack_agents
    balanced_cnt = args.combined_agents

    ppo_cnt = args.ppo_agents

    dqn_cnt = args.dqn_agents
    buffer_cap = args.replay_buffer_capacity

    net_arch = tuple(ast.literal_eval('(' + args.net_arch + ')'))
    normalize = not args.unnormalize
    config_dict['normalize'] = normalize

    ident = (args.ident +
             '_agent_count_'
             '_rand_' + str(rand_cnt) +
             '_defend_' + str(defend_cnt) +
             '_attack_' + str(attack_cnt) +
             '_combined_' + str(balanced_cnt) +
             '_ppo_' + str(ppo_cnt) +
             '_dqn_' + str(dqn_cnt) +
             '_' +
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


    non_train_dict = {DICT_TRAIN: False,
                      DICT_CLONABLE: False,
                      DICT_CLONE_REPLACABLE: False,
                      DICT_MUTATION_REPLACABLE: False,
                      DICT_IS_WORKER: False,
                      }
    create_rand = lambda i, env: (RandPolicy(env.action_space), non_train_dict.copy())
    create_defend = lambda i, env: (DefendPolicy(agent_id=0,
                                                 team='red',
                                                 mode="easy",
                                                 flag_keepout=config_dict['flag_keepout'],
                                                 catch_radius=config_dict["catch_radius"],
                                                 using_pyquaticus=True,
                                                 ), non_train_dict.copy()
                                    )
    create_attack = lambda i, env: (AttackPolicy(agent_id=0,
                                                 mode="easy",
                                                 using_pyquaticus=True,
                                                 ), non_train_dict.copy()
                                    )
    create_balance = lambda i, env: (BalancedPolicy(agent_id=0,
                                                    team='red',
                                                    mode="easy",
                                                    flag_keepout=config_dict['flag_keepout'],
                                                    catch_radius=config_dict["catch_radius"],
                                                    using_pyquaticus=True,
                                                    defensiveness=20.,
                                                    ), non_train_dict.copy()
                                     )

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
    non_learning_sizes = [rand_cnt,
                          defend_cnt,
                          attack_cnt,
                          balanced_cnt,
                          ]
    non_lerning_construct = [create_rand,
                             create_defend,
                             create_attack,
                             create_balance,
                             ]
    if args.split_learners:
        pop_sizes = non_learning_sizes + [ppo_cnt,
                                          dqn_cnt,
                                          ]

        worker_constructors = non_lerning_construct + [create_ppo,
                                                       create_dqn,
                                                       ]
    else:
        pop_sizes = non_learning_sizes + [ppo_cnt + dqn_cnt]


        def create_learner(i, env):
            if i < ppo_cnt:
                return create_ppo(i, env)
            else:
                return create_dqn(i - ppo_cnt, env)


        worker_constructors = non_lerning_construct + [create_learner]

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
        elif 'WrappedPolicy' in str(type(animal)):
            return animal.identity
        else:
            return 'random'


    if args.display:
        elos = trainer.classic_elos.numpy().copy()
        worst = np.argmin(elos)

        best = np.argmax(elos)
        elos[best] = -np.inf
        second_best = np.argmax(elos)
        print(worst, best, second_best)
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
                elo_tracker = dict()
                for i in range(sum(pop_sizes)):
                    identity = typer(i)
                    if identity not in elo_tracker:
                        elo_tracker[identity] = []
                    elo_tracker[identity].append(i)
                print('all elos')
                for identity in elo_tracker:
                    print('\telo of', identity, 'agents:', classic_elos[elo_tracker[identity]])

                print('avg elos')
                for identity in elo_tracker:
                    print('\tavgelo of', identity, 'agents:', np.mean(classic_elos[elo_tracker[identity]]))

            if not (trainer.info['epochs'])%args.ckpt_freq:
                print('saving')
                trainer.save(save_dir)
                print('done saving')
            print('time', time.time() - tim)
            print()

    trainer.clear()
    import shutil
    shutil.rmtree(data_folder)