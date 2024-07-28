import argparse, os, sys
from repos.pyquaticus.pyquaticus import pyquaticus_v0
import repos.pyquaticus.pyquaticus.utils.rewards as rew
from src.coevolver import PettingZooCaptianCoevolution
from src.game_outcome import PettingZooOutcomeFn
from src.utils.dict_keys import (DICT_IS_WORKER,
                                 DICT_TRAIN,
                                 DICT_CLONABLE,
                                 DICT_CLONE_REPLACABLE,
                                 DICT_MUTATION_REPLACABLE,
                                 )
from multi_agent_algs.ppo.PPO import WorkerPPO
from stable_baselines3.ppo import MlpPolicy
from experiments.pyquaticus_coevolution import env_constructor, CTFOutcome, RandPolicy




DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

print(DIR)
data_folder = os.path.join(DIR, 'data', 'basic_ppo_against_easy')

PARSER = argparse.ArgumentParser()

PARSER.add_argument('--render', action='store_true', required=False,
                    help="Enable rendering")

args = PARSER.parse_args()

RENDER_MODE = 'human' if args.render else None

trainer = PettingZooCaptianCoevolution(population_sizes=[1,
                                                         1
                                                         ],
                                       outcome_fn=CTFOutcome(),
                                       env_constructor=lambda: env_constructor(render_mode=RENDER_MODE),
                                       worker_constructors=[
                                           lambda i, env: (WorkerPPO(policy=MlpPolicy,
                                                                     env=env,
                                                                     #batch_size=100,
                                                                     #n_steps=100,
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

save_dir = os.path.join(DIR, 'data', 'save', 'test_ppo_coevloution')
if os.path.exists(save_dir):
    trainer.load(save_dir=save_dir)
for episode in range(1000):
    print('starting epoch', trainer.info['epochs'])
    trainer.epoch()
    if not (episode+1)%10:
        print('saving')
        trainer.save(save_dir)
        print('done saving')
trainer.clear()
