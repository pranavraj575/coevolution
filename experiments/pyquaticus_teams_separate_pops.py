if __name__ == '__main__':
    import argparse, os, sys, ast
    from experiments.pyquaticus_utils.arg_parser import *

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))

    PARSER = argparse.ArgumentParser()
    add_team_args(PARSER)

    PARSER.add_argument('--rand-agents', type=int, required=False, default=0,
                        help="number of random agents to use")

    PARSER.add_argument('--defend-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    PARSER.add_argument('--attack-agents', type=int, required=False, default=0,
                        help="number of random agents to use")
    add_learning_agent_args(PARSER)
    add_pyquaticus_args(PARSER)
    add_experiment_args(PARSER, 'pyquaticus_teams_separate_pops')
    add_coevolution_args(PARSER)

    args = PARSER.parse_args()
    if not args.unblock_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    team_size = args.team_size
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

    import time
    import numpy as np
    import random

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_dict = config_dict_std
    config_dict["max_screen_size"] = (float('inf'), float('inf'))
    # config_dict["world_size"] = [160.0, 80.0]
    config_dict["world_size"] = ast.literal_eval('(' + args.arena_size + ')')
    # config_dict['tag_on_wall_collision']=True
    reward_config = {i: custom_rew2 for i in range(team_size*2)}  # Example Reward Config

    test_env = pyquaticus_v0.PyQuaticusEnv(render_mode=None,
                                           team_size=team_size,
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
             '_team_size_' + str(team_size) +
             '_arena_size_' + str(args.arena_size.replace('.', '_').replace(',', '__')) +
             '_agent_count_' +
             (('_rand_' + str(rand_cnt)) if rand_cnt else '') +
             (('_defend_' + str(defend_cnt)) if defend_cnt else '') +
             (('_attack_' + str(attack_cnt)) if attack_cnt else '') +
             (('_net_arch_' + '_'.join([str(s) for s in net_arch])) if ppo_cnt + dqn_cnt else '') +
             (('_ppo_' + str(ppo_cnt)) if ppo_cnt else '') +
             (('_dqn_' + str(dqn_cnt)) if dqn_cnt else '') +
             '_' +
             (('_replay_buffer_capacity_' + str(buffer_cap)) if dqn_cnt else '') +
             '_protect_new_' + str(args.protect_new) +
             '_mutation_prob_' + str(args.mutation_prob).replace('.', '_') +
             ('_clone_replacments_' + str(clone_replacements) if clone_replacements is not None else '') +
             ('_dont_normalize_obs' if not normalize else '')
             )

    data_folder = os.path.join(DIR, 'data', 'temp', ident)
    save_dir = os.path.join(DIR, 'data', 'save', ident)


    def env_constructor(train_infos):
        return pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE,
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
    train_info_dict = {DICT_TRAIN: False,
                       DICT_COLLECT_ONLY: True,
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
    non_lerning_construct = [create_rand,
                             create_defend,
                             create_attack,
                             ]

    agent_counts = [rand_cnt,
                    defend_cnt,
                    attack_cnt,
                    ppo_cnt,
                    dqn_cnt
                    ]


    def create_agent(i, env):
        for agent_cnt, creation in ((rand_cnt, create_rand),
                                    (defend_cnt, create_defend),
                                    (attack_cnt, create_attack),
                                    (ppo_cnt, create_ppo),
                                    (dqn_cnt, create_dqn)
                                    ):
            if i < agent_cnt:
                return creation(i, env)
            i -= agent_cnt


    if sum(agent_counts) == 0:
        raise Exception("no agents specified, at least one of --*-agents must be greater than 0")
    max_cores = len(os.sched_getaffinity(0))
    if args.display:
        proc = 0
    else:
        proc = args.processes
    trainer = PettingZooCaptianCoevolution(population_sizes=[sum(agent_counts) for _ in range(team_size)],
                                           team_sizes=(team_size, team_size),
                                           outcome_fn_gen=CTFOutcome,
                                           env_constructor=env_constructor,
                                           worker_constructors=[create_agent for _ in range(team_size)],
                                           storage_dir=data_folder,
                                           protect_new=args.protect_new,
                                           processes=proc,
                                           # draw member i from population i
                                           # i.e. each team has team_size members drawn from team_size populations
                                           member_to_population=lambda team_idx, member_idx: {member_idx},
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
        worst = [np.argmin(elos[i*sum(agent_counts):(i + 1)*sum(agent_counts)]) for i in range(team_size)]

        best = [np.argmax(elos[i*sum(agent_counts):(i + 1)*sum(agent_counts)]) for i in range(team_size)]
        elos[best] = -np.inf
        second_best = [np.argmax(elos[i*sum(agent_counts):(i + 1)*sum(agent_counts)]) for i in range(team_size)]

        classic_elos = trainer.classic_elos.numpy()
        idents_and_elos = []
        for i in range(sum(agent_counts)*team_size):
            idents_and_elos.append((typer(i), classic_elos[i]))
        print('all elos by index')
        for i, animal in enumerate(test_animals):
            ip = i - len(test_animals)
            print(ip, ' (', typer(ip), '): ', None, sep='')
        for i, (identity, elo) in enumerate(idents_and_elos):
            if not i%sum(agent_counts):
                print("POPULATION BOUNDARY")
            print(i, ' (', identity, '): ', elo, sep='')
        if idxs is None:

            print('best agents have elo', trainer.classic_elos[best],
                  'and are types', [typer(idx) for idx in best])
            print('second best agents have elo', trainer.classic_elos[second_best],
                  'and are types', [typer(idx) for idx in second_best])
            print('worst agents have elo', trainer.classic_elos[worst],
                  'and are type', [typer(idx) for idx in worst])

            print('playing worst (blue, ' + str([typer(idx) for idx in worst])
                  + ') against best (red, ' + str([typer(idx) for idx in best]) + ')')

            outcom = CTFOutcome()
            agents = []
            for team in worst, best:
                m = []
                for idx in team:
                    agent = trainer.load_animal(trainer.index_to_pop_index(idx))[0]
                    agent.policy.set_training_mode(False)
                    m.append(agent)
                agents.append(m)

            outcom.get_outcome(
                team_choices=[torch.tensor(worst), torch.tensor(best)],
                agent_choices=agents,
                env=env_constructor(None),
                updated_train_infos=[[non_train_dict]*team_size]*2,
            )

            print('playing second best (blue, ' + str([typer(idx) for idx in second_best])
                  + ') against best (red, ' + str([typer(idx) for idx in best]) + ')')

            agents = []
            for team in second_best, best:
                m = []
                for idx in team:
                    agent = trainer.load_animal(trainer.index_to_pop_index(idx))[0]
                    agent.policy.set_training_mode(False)
                    m.append(agent)
                agents.append(m)

            outcom.get_outcome(
                team_choices=[torch.tensor(second_best), torch.tensor(best)],
                agent_choices=agents,
                env=env_constructor(None),
                updated_train_infos=[[non_train_dict]*team_size]*2,
            )

        else:
            A, B = [ast.literal_eval('(' + team + ')') for team in idxs.split(';')]

            print('playing second best (blue, ' + str([typer(idx) for idx in B])
                  + ') against best (red, ' + str([typer(idx) for idx in A]) + ')')
            agents = []

            for team in B, A:
                m = []
                for idx in team:
                    if idx >= 0:
                        agent = trainer.load_animal(trainer.index_to_pop_index(idx))[0]
                        agent.policy.set_training_mode(False)
                    else:
                        agent = test_animals[idx][0]
                    m.append(agent)
                agents.append(m)

            outcom = CTFOutcome()
            outcom.get_outcome(
                team_choices=[torch.tensor(B), torch.tensor(A)],
                agent_choices=agents,
                env=env_constructor(None),
                updated_train_infos=[[non_train_dict]*team_size]*2,
            )
    else:
        while trainer.epochs < args.epochs:
            tim = time.time()
            print('starting epoch', trainer.info['epochs'], 'at time', time.strftime('%H:%M:%S'))
            epoch_info = trainer.epoch()
            if True:
                id_to_idxs = dict()
                for i in range(sum(agent_counts)*team_size):
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
                print('saving')
                trainer.save(save_dir)
                print('done saving')
            print('time', time.time() - tim)
            print()

    trainer.clear()
    import shutil

    shutil.rmtree(data_folder)
