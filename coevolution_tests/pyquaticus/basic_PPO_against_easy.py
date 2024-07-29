import argparse, os, sys
from repos.pyquaticus.pyquaticus import pyquaticus_v0
from src.coevolver import PettingZooCaptianCoevolution
from src.utils.dict_keys import (DICT_IS_WORKER,
                                 DICT_TRAIN,
                                 DICT_CLONABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_MUTATION_REPLACABLE,
                                 )
from unstable_baselines3.ppo.PPO import WorkerPPO
from stable_baselines3.ppo import MlpPolicy
from experiments.pyquaticus_coevolution import reward_config, config_dict, CTFOutcome, RandPolicy

DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

data_folder = os.path.join(DIR, 'data', 'basic_ppo_against_easy')

PARSER = argparse.ArgumentParser()

PARSER.add_argument('--render', action='store_true', required=False,
                    help="Enable rendering")

args = PARSER.parse_args()

RENDER_MODE = 'human' if args.render else None


def env_constructor(train_infos):
    return pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE,
                                       reward_config=reward_config,
                                       team_size=1,
                                       config_dict=config_dict,
                                       )


trainer = PettingZooCaptianCoevolution(population_sizes=[1,
                                                         1
                                                         ],
                                       outcome_fn_gen=CTFOutcome,
                                       env_constructor=env_constructor,
                                       worker_constructors=[
                                           lambda i, env: (WorkerPPO(policy=MlpPolicy,
                                                                     env=env,
                                                                     # batch_size=100,
                                                                     # n_steps=100,
                                                                     policy_kwargs={
                                                                         'net_arch': dict(pi=[128, 64],
                                                                                          vf=[128, 64])
                                                                     }
                                                                     ),
                                                           {DICT_TRAIN: True,
                                                            DICT_CLONABLE: False,
                                                            DICT_CLONE_REPLACABLE: False,
                                                            DICT_MUTATION_REPLACABLE: False,
                                                            DICT_IS_WORKER: True,
                                                            }
                                                           ),
                                           lambda i, env: (RandPolicy(env.action_space), {DICT_TRAIN: False,
                                                                                          DICT_CLONABLE: False,
                                                                                          DICT_CLONE_REPLACABLE: False,
                                                                                          DICT_MUTATION_REPLACABLE: False,
                                                                                          DICT_IS_WORKER: False,
                                                                                          }),
                                       ],
                                       zoo_dir=os.path.join(data_folder, 'zoo'),

                                       # ppo is always on the first team, random always second
                                       member_to_population=lambda team_idx, member_idx: {team_idx},
                                       )

save_dir = os.path.join(DIR, 'data', 'save', 'basic_ppo_against_easy')
if os.path.exists(save_dir):
    trainer.load(save_dir=save_dir)

while trainer.epochs < 3030:
    print('starting epoch', trainer.info['epochs'])
    trainer.epoch()

    print('elos:', trainer.get_classic_elo(1000))
    if not (trainer.info['epochs'])%10:
        print('saving')
        trainer.save(save_dir)
        print('done saving')
    print()


def env_constructor2(train_infos):
    return pyquaticus_v0.PyQuaticusEnv(render_mode='human',
                                       reward_config=reward_config,
                                       team_size=1,
                                       config_dict=config_dict,
                                       )


trainer.env_constructor = env_constructor2
trainer.epoch()
trainer.clear()
