import argparse, os, sys, ast, shutil
import torch.random
from unstable_baselines3 import WorkerPPO, WorkerDQN
from unstable_baselines3.dqn import MlpPolicy as DQNMlp
from unstable_baselines3.ppo import MlpPolicy as PPOMlp

from repos.pyquaticus.pyquaticus import pyquaticus_v0
from repos.pyquaticus.pyquaticus.config import config_dict_std
from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker
from experiments.pyquaticus_utils.reward_fns import custom_rew, custom_rew2, RandPolicy
from experiments.pyquaticus_utils.outcomes import CTFOutcome
from experiments.pyquaticus_utils.wrappers import policy_wrapper

from src.coevolver import PettingZooCaptianCoevolution
from src.utils.dict_keys import *

config_dict = config_dict_std
config_dict["max_screen_size"] = (float('inf'), float('inf'))
# reset later
config_dict["sim_speedup_factor"] = 40
config_dict["max_time"] = 420.
# config_dict['tag_on_wall_collision']=True
reward_config = {0: custom_rew2, 1: custom_rew2, 5: None}  # Example Reward Config

if __name__ == '__main__':
    import time
    import numpy as np
    import random

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--rand-agents', type=int, required=False, default=0,
                        help="number of random agents to use")

    PARSER.add_argument('--defend-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    PARSER.add_argument('--attack-agents', type=int, required=False, default=0,
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

    PARSER.add_argument('--protect-new', type=int, required=False, default=500,
                        help="protect new agents for this number of breeding epochs")
    PARSER.add_argument('--mutation-prob', type=float, required=False, default=0.,
                        help="probabality of mutating agents each epoch (should probably be very small)")
    PARSER.add_argument('--clone-replacements', type=int, required=False, default=None,
                        help="number of agents to try replacing each epoch (default all)")

    PARSER.add_argument('--max-time', type=float, required=False, default=420.,
                        help="max sim time of each episode")
    PARSER.add_argument('--sim-speedup-factor', type=int, required=False, default=40,
                        help="skips frames to speed up episodes")
    PARSER.add_argument('--unnormalize', action='store_true', required=False,
                        help="do not normalize, arg is necessary to use some pyquaticus bots")

    PARSER.add_argument('--epochs', type=int, required=False, default=5000,
                        help="epochs to train for")
    PARSER.add_argument('--processes', type=int, required=False, default=0,
                        help="number of processes to use")

    PARSER.add_argument('--reset', action='store_true', required=False,
                        help="do not load from save")
    PARSER.add_argument('--ckpt_freq', type=int, required=False, default=25,
                        help="checkpoint freq")
    PARSER.add_argument('--dont-backup', action='store_true', required=False,
                        help="do not backup a copy of previous save")
    PARSER.add_argument('--ident', action='store', required=False, default='pyquaticus_coevolution',
                        help='identification to add to folder')

    PARSER.add_argument('--display', action='store_true', required=False,
                        help="skip training and display saved model")

    PARSER.add_argument('--idxs-to-display', action='store', required=False, default=None,
                        help='which agent indexes to display, in the format "i,j" (used with --display)')

    PARSER.add_argument('--render', action='store_true', required=False,
                        help="Enable rendering")
    PARSER.add_argument('--seed', type=int, required=False, default=0,
                        help="random seed")

    PARSER.add_argument('--unblock-gpu', action='store_true', required=False,
                        help="unblock using gpu ")
    args = PARSER.parse_args()
    if not args.unblock_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    test_env = pyquaticus_v0.PyQuaticusEnv(render_mode=None,
                                           team_size=1,
                                           config_dict=config_dict,
                                           )
    obs_normalizer = test_env.agent_obs_normalizer
    DefendPolicy = policy_wrapper(BaseDefender, agent_obs_normalizer=obs_normalizer, identity='def')
    AttackPolicy = policy_wrapper(BaseAttacker, agent_obs_normalizer=obs_normalizer, identity='att')

    config_dict["sim_speedup_factor"] = args.sim_speedup_factor
    config_dict["max_time"] = args.max_time
    RENDER_MODE = 'human' if args.render or args.display else None

    clone_replacements = args.clone_replacements

    rand_cnt = args.rand_agents

    defend_cnt = args.defend_agents
    attack_cnt = args.attack_agents

    ppo_cnt = args.ppo_agents

    dqn_cnt = args.dqn_agents
    buffer_cap = args.replay_buffer_capacity

    net_arch = tuple(ast.literal_eval('(' + args.net_arch + ')'))
    normalize = not args.unnormalize
    config_dict['normalize'] = normalize

    ident = (args.ident +
             '_agent_count_' +
             (('_rand_' + str(rand_cnt)) if rand_cnt else '') +
             (('_defend_' + str(defend_cnt)) if defend_cnt else '') +
             (('_attack_' + str(attack_cnt)) if attack_cnt else '') +
             (('_net_arch_' + '_'.join([str(s) for s in net_arch])) if ppo_cnt + dqn_cnt else '') +
             (('_ppo_' + str(ppo_cnt)) if ppo_cnt else '') +
             (('_dqn_' + str(dqn_cnt)) if dqn_cnt else '') +
             '_' +
             (('_replay_buffer_capacity_' + str(buffer_cap)) if dqn_cnt else '') +
             ('_split_learners_' if args.split_learners and ppo_cnt and dqn_cnt else '') +
             '_protect_new_' + str(args.protect_new) +
             '_mutation_prob_' + str(args.mutation_prob).replace('.', '_') +
             ('_clone_replacments_' + str(clone_replacements) if clone_replacements is not None else '') +
             ('_dont_normalize_obs' if not normalize else '')
             )

    data_folder = os.path.join(DIR, 'data', 'temp', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)
    backup_dir = os.path.join(DIR, 'data', 'save', 'backups', ident)


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
                                                 mode="hard",
                                                 flag_keepout=config_dict['flag_keepout'],
                                                 catch_radius=config_dict["catch_radius"],
                                                 using_pyquaticus=True,
                                                 ), non_train_dict.copy()
                                    )
    create_attack = lambda i, env: (AttackPolicy(agent_id=0,
                                                 mode="hard",
                                                 using_pyquaticus=True,
                                                 ), non_train_dict.copy()
                                    )
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
    non_learning_sizes = [rand_cnt,
                          defend_cnt,
                          attack_cnt,
                          ]
    non_lerning_construct = [create_rand,
                             create_defend,
                             create_attack,
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
    if sum(pop_sizes) == 0:
        raise Exception("no agents specified, at least one of --*-agents must be greater than 0")
    max_cores = len(os.sched_getaffinity(0))
    if args.display:
        proc = 0
    else:
        proc = args.processes
    trainer = PettingZooCaptianCoevolution(population_sizes=pop_sizes,
                                           outcome_fn_gen=CTFOutcome,
                                           env_constructor=env_constructor,
                                           worker_constructors=worker_constructors,
                                           storage_dir=data_folder,
                                           protect_new=args.protect_new,
                                           processes=proc,
                                           # member_to_population=lambda team_idx, member_idx: {team_idx},
                                           # for some reason this overesetimates by a factor of 3, so we fix it
                                           max_steps_per_ep=(
                                                   1 + (1/3)*config_dict['render_fps']*config_dict['max_time']/
                                                   config_dict['sim_speedup_factor']),
                                           mutation_prob=args.mutation_prob,
                                           clone_replacements=clone_replacements,
                                           )
    if not args.reset and os.path.exists(save_dir):
        print('loading from', save_dir)
        trainer.load(save_dir=save_dir)
    print('seting save directory as', save_dir)
    if args.display:
        from unstable_baselines3.common import DumEnv

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
        if idxs is None:

            print('best agent has elo', trainer.classic_elos[best], 'and is type', typer(best))
            print('second best agent has elo', trainer.classic_elos[second_best], 'and is type', typer(second_best))
            print('worst agent has elo', trainer.classic_elos[worst], 'and is type', typer(worst))
            print('playing worst (blue, ' + typer(worst) + ') against best (red, ' + typer(best) + ')')

            outcom = CTFOutcome()
            agents = (trainer.load_animal(trainer.index_to_pop_index(worst))[0],
                      trainer.load_animal(trainer.index_to_pop_index(best))[0])
            for agent in agents:
                agent.policy.set_training_mode(False)
            outcom.get_outcome(
                team_choices=[[torch.tensor(worst)], [torch.tensor(best)]],
                agent_choices=[[agents[0]], [agents[1]]],
                env=env_constructor(None),
                updated_train_infos=[[non_train_dict],
                                     [non_train_dict]]
            )
            print('playing second best (blue, ' + typer(second_best) + ') against best (red, ' + typer(best) + ')')

            agents = (trainer.load_animal(trainer.index_to_pop_index(second_best))[0],
                      trainer.load_animal(trainer.index_to_pop_index(best))[0])
            for agent in agents:
                agent.policy.set_training_mode(False)

            outcom.get_outcome(
                team_choices=[[torch.tensor(second_best)], [torch.tensor(best)]],
                agent_choices=[[agents[0]], [agents[1]]],
                env=env_constructor(None),
                updated_train_infos=[[non_train_dict],
                                     [non_train_dict]]
            )
        else:
            i, j = ast.literal_eval('(' + idxs + ')')
            print('playing', i, '(blue, ' + typer(i) + ') against', j, '(red, ' + typer(j) + ')')
            agents = []
            for k, idx in enumerate((i, j)):
                if idx >= 0:
                    agent = trainer.load_animal(trainer.index_to_pop_index(idx))[0]
                    agent.policy.set_training_mode(False)
                    agents.append(agent)
                else:
                    agents.append(test_animals[idx][0])
            outcom = CTFOutcome()
            outcom.get_outcome(
                team_choices=[[torch.tensor(i)], [torch.tensor(j)]],
                agent_choices=[[agents[0]], [agents[1]]],
                env=env_constructor(None),
                updated_train_infos=[[non_train_dict],
                                     [non_train_dict]]
            )
    else:
        while trainer.epochs < args.epochs:
            tim = time.time()
            print('starting epoch', trainer.info['epochs'], 'at time', time.strftime('%H:%M:%S'))
            epoch_info = trainer.epoch()
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

            if not (trainer.info['epochs'])%args.ckpt_freq:
                if not args.dont_backup and os.path.exists(save_dir):
                    print('backing up')
                    shutil.copytree(save_dir, backup_dir)
                print('saving')
                trainer.save(save_dir)
                print('done saving')
            print('time', time.time() - tim)
            print()

    trainer.clear()
    import shutil

    shutil.rmtree(data_folder)
