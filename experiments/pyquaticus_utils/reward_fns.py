

DEBUG_MESSAGES = False


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

    flag_capture_rew = 1.
    flag_pickup_rew = 1.
    tag_reward = .05
    oob_penalty = 1.

    # Penalize player for opponent grabbing team flag
    if params["opponent_flag_pickup"] and not prev_params["opponent_flag_pickup"]:
        reward += -flag_pickup_rew
        if DEBUG_MESSAGES:
            print('enemey flag pickup')
    # Reward player for grabbing opponents flag
    if params["has_flag"] and not prev_params['has_flag']:
        if DEBUG_MESSAGES:
            print('flag pickup')
        reward += flag_pickup_rew

    # penalize player for dropping flag
    if not params["team_flag_capture"] and (prev_params["has_flag"] and not params['has_flag']):
        if DEBUG_MESSAGES:
            print('dropped flag')
        reward += -flag_pickup_rew

    # Penalize player for opponent successfully capturing team flag
    if params["opponent_flag_capture"] and not prev_params["opponent_flag_capture"]:
        if DEBUG_MESSAGES:
            print('enemy flag cap')
        reward += -flag_capture_rew
    # Reward player for capturing opponents flag
    if params["team_flag_capture"] and not prev_params["team_flag_capture"]:
        if DEBUG_MESSAGES:
            print('flag cap')
        reward += flag_capture_rew

    # Check to see if agent was tagged
    if params["agent_tagged"][params["agent_id"]] and not prev_params["agent_tagged"][params["agent_id"]]:
        if DEBUG_MESSAGES:
            print('agent tagged')
        reward += -tag_reward
    # Check to see if agent tagged an opponent
    tagged_opponent = params["agent_captures"][params["agent_id"]]
    if tagged_opponent is not None:
        reward += tag_reward
        if DEBUG_MESSAGES:
            print('opponent tagged')
        if prev_params["opponent_" + str(tagged_opponent) + "_has_flag"]:
            reward += flag_pickup_rew

    # Penalize agent if it went out of bounds (Hit border wall)
    if params["agent_oob"][params["agent_id"]] == 1:
        reward += -oob_penalty
    if reward != 0:
        if DEBUG_MESSAGES:
            print(reward)
    return reward



class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs, *args, **kwargs):
        return self.action_space.sample()
