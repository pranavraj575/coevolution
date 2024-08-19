import argparse, os, sys, ast, shutil
import torch.random
from unstable_baselines3 import WorkerPPO, WorkerDQN
from unstable_baselines3.dqn import MlpPolicy as DQNMlp
from unstable_baselines3.ppo import MlpPolicy as PPOMlp

from repos.pyquaticus.pyquaticus.config import config_dict_std
from repos.pyquaticus.pyquaticus.base_policies.base_defend import BaseDefender
from repos.pyquaticus.pyquaticus.base_policies.base_attack import BaseAttacker
from experiments.pyquaticus_utils.reward_fns import custom_rew, custom_rew2, RandPolicy
from experiments.pyquaticus_utils.outcomes import CTFOutcome
from experiments.pyquaticus_utils.wrappers import policy_wrapper, MyQuaticusEnv

from src.coevolver import PettingZooCaptianCoevolution
from src.utils.dict_keys import *

reward_config = {0: custom_rew2, 1: custom_rew2, 5: None}  # Example Reward Config

if __name__ == '__main__':
    import time, numpy as np, random
    from experiments.pyquaticus_utils.arg_parser import *

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--rand-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    PARSER.add_argument('--defend-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    PARSER.add_argument('--attack-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    add_learning_agent_args(PARSER, split_learners=True)
    add_coevolution_args(PARSER)
    add_pyquaticus_args(PARSER, arena_size=False, flag_keepout=False)
    add_experiment_args(PARSER, 'pyquaticus_coevolution')
    PARSER.add_argument('--dont-backup', action='store_true', required=False,
                        help="do not backup a copy of previous save")

    args = PARSER.parse_args()
    if not args.unblock_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_dict = config_dict_std
    update_config_dict(config_dict, args)

    test_env = MyQuaticusEnv(render_mode=None,
                             team_size=1,
                             config_dict=config_dict,
                             )
    obs_normalizer = test_env.agent_obs_normalizer
    DefendPolicy = policy_wrapper(BaseDefender, agent_obs_normalizer=obs_normalizer, identity='def')
    AttackPolicy = policy_wrapper(BaseAttacker, agent_obs_normalizer=obs_normalizer, identity='att')

    RENDER_MODE = get_render_mode(args)

    clone_replacements = args.clone_replacements

    rand_cnt = args.rand_agents

    defend_cnt = args.defend_agents
    attack_cnt = args.attack_agents

    ppo_cnt = args.ppo_agents

    dqn_cnt = args.dqn_agents
    buffer_cap = args.replay_buffer_capacity

    net_arch = tuple(ast.literal_eval('(' + args.net_arch + ')'))

    ident = (args.ident +
             pyquaticus_string(args) +
             '_agent_count_' +
             (('_rand_' + str(rand_cnt)) if rand_cnt else '') +
             (('_def_' + str(defend_cnt)) if defend_cnt else '') +
             (('_att_' + str(attack_cnt)) if attack_cnt else '') +
             learning_agents_string(args) +
             coevolution_string(args)
             )

    data_folder = os.path.join(DIR, 'data', 'temp', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)
    backup_dir = os.path.join(DIR, 'data', 'save', 'backups', ident)


    def env_constructor(train_infos):
        return MyQuaticusEnv(save_video=args.save_video is not None,
                             render_mode=RENDER_MODE,
                             reward_config=reward_config,
                             team_size=1,
                             config_dict=config_dict,
                             frame_freq=args.frame_freq,
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
                                           protect_elite=args.elite_protection,
                                           )
    if not args.reset and os.path.exists(save_dir):
        print('loading from', save_dir)
        trainer.load(save_dir=save_dir)
    print('seting save directory as', save_dir)
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
        if idxs is None:

            print('best agent has elo', trainer.classic_elos[best], 'and is type', typer(best))
            print('second best agent has elo', trainer.classic_elos[second_best], 'and is type', typer(second_best))
            print('worst agent has elo', trainer.classic_elos[worst], 'and is type', typer(worst))
            print('playing best (blue, ' + typer(best) + ') against worst (red, ' + typer(worst) + ')')

            outcom = CTFOutcome()
            agents = (trainer.load_animal(trainer.index_to_pop_index(best))[0],
                      trainer.load_animal(trainer.index_to_pop_index(worst))[0])
            for agent in agents:
                agent.policy.set_training_mode(False)
            outcom.get_outcome(
                team_choices=[[torch.tensor(best)], [torch.tensor(worst)]],
                agent_choices=[[agents[0]], [agents[1]]],
                env=env,
                updated_train_infos=[[non_train_dict],
                                     [non_train_dict]]
            )
            print('playing best (blue, ' + typer(best) + ') against second best (red, ' + typer(second_best) + ')')

            agents = (trainer.load_animal(trainer.index_to_pop_index(best))[0],
                      trainer.load_animal(trainer.index_to_pop_index(second_best))[0])
            for agent in agents:
                agent.policy.set_training_mode(False)

            outcom.get_outcome(
                team_choices=[[torch.tensor(best)], [torch.tensor(second_best)]],
                agent_choices=[[agents[0]], [agents[1]]],
                env=env,
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
                env=env,
                updated_train_infos=[[non_train_dict],
                                     [non_train_dict]]
            )
        if args.save_video is not None:
            env.write_video(video_file=args.save_video)
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
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)
                    shutil.copytree(save_dir, backup_dir)
                    print('done backing up')
                print('saving')
                trainer.save(save_dir)
                print('done saving')
            print('time', time.time() - tim)
            print()

    trainer.clear()

    shutil.rmtree(data_folder)
