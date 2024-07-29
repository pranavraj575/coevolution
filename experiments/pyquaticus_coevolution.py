import argparse, os, sys
from repos.pyquaticus.pyquaticus import pyquaticus_v0
from repos.pyquaticus.pyquaticus.config import config_dict_std

from src.game_outcome import PettingZooOutcomeFn
from unstable_baselines3.unstable_baselines3.common.better_multi_alg import multi_agent_algorithm


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
    # Penalize player for opponent grabbing team flag
    if params["opponent_flag_pickup"] and not prev_params["opponent_flag_pickup"]:
        reward += -.2
    # Penalize player for opponent successfully capturing team flag
    if params["opponent_flag_capture"] and not prev_params["opponent_flag_capture"]:
        reward += -1

    # Reward player for grabbing opponents flag
    if params["has_flag"] and not prev_params['has_flag']:
        reward += .2

    # penalize player for dropping flag
    if not params["team_flag_capture"] and (prev_params["has_flag"] and not params['has_flag']):
        reward += - .2

    # Reward player for capturing opponents flag
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        reward += 1
    # Check to see if agent was tagged
    if params["agent_tagged"][params["agent_id"]] and not prev_params["agent_tagged"][params["agent_id"]]:
        if prev_params["has_flag"]:
            reward += -.05
        else:
            reward += -.02

    # Check to see if agent tagged an opponent
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        if prev_params["opponent_" + str(tagged_opponent) + "_has_flag"]:
            reward += .03
        else:
            reward += .03
    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward -= 1
    return reward


class CTFOutcome(PettingZooOutcomeFn):
    def _get_outcome_from_agents(self, agent_choices, index_choices, updated_train_infos, env):
        agent_choices = agent_choices[0] + agent_choices[1]
        updated_train_infos = updated_train_infos[0] + updated_train_infos[1]

        # env is set up so the first k agents are team blue and the last k agents are team red
        alg = multi_agent_algorithm(env=env,
                                    workers={i: agent_choices[i] for i in range(len(agent_choices))},
                                    worker_infos={i: updated_train_infos[i] for i in range(len(agent_choices))},
                                    )
        alg.learn(total_timesteps=10000,
                  number_of_eps=1,
                  )
        score = (env.unwrapped.game_score['blue_captures'], env.unwrapped.game_score['red_captures'])
        print(score)
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


config_dict = config_dict_std
config_dict["max_screen_size"] = (float('inf'), float('inf'))
config_dict["max_time"] = 420.
config_dict["sim_speedup_factor"] = 40
# config_dict['tag_on_wall_collision']=True
reward_config = {0: custom_rew2, 1: None, 5: None}  # Example Reward Config

if __name__ == '__main__':
    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))))

    data_folder = os.path.join(DIR, 'data', 'pyquaticus_coevolution')

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--render', action='store_true', required=False,
                        help="Enable rendering")
    PARSER.add_argument('--max-time', type=float, required=False, default=420.,
                        help="max sim time of each episode")
    PARSER.add_argument('--sim-speedup-factor', type=int, required=False, default=40,
                        help="skips frames to speed up episodes")

    args = PARSER.parse_args()
    config_dict["sim_speedup_factor"] = args.sim_speedup_factor
    config_dict["max_time"] = args.max_time
    RENDER_MODE = 'human' if args.render else None


    def env_constructor():
        return pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE,
                                           reward_config=reward_config,
                                           team_size=1,
                                           config_dict=config_dict,
                                           )
