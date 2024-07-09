import pickle
import os, sys
import time

import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer
from ray.rllib.policy.policy import Policy

# Use our example multi-agent CartPole environment to train in.
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole

# Set up a multi-agent Algorithm, training two policies independently.
dummy_policies = ['dummy1', 'dummy2']
policies = dummy_policies + ['pol' + str(i) for i in range(10)]

policies_to_use = [('pol3', True), ('pol3', False)]
# which to put in for agents 0 and 1, and whether to train them
# if False, the policy will be used, it will be fixed, and the replay buffer will not be updated

reset_replay = []
# reset_replay = ['pol0']
reset_all = False

# only train the correct policies
policies_to_train = [thing for thing, b in policies_to_use if b]


def pol_map(agent_id, episode, worker, **kw):
    # policies_to_use[id] is the correct policy
    # however, if we are holding any fixed, we map it to the appropriate dummy
    thing, to_train = policies_to_use[agent_id]
    if to_train:
        return thing
    else:
        return dummy_policies[agent_id]


DIR = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))

save_dir = os.path.join(DIR, '..', 'data', 'TEST')
if not os.path.exists(save_dir) or reset_all:
    my_ma_config = DQNConfig().multi_agent(
        # Which policies should RLlib create and train?
        # initialize all policies, only train the important ones
        policies=set(policies),
        # Let RLlib know, which agents in the environment (we'll have "agent1"
        # and "agent2") map to which policies.
        policy_mapping_fn=pol_map,

        policies_to_train=policies_to_train,
    )
    my_ma_config.training(
        replay_buffer_config={'type': MultiAgentReplayBuffer}
    )

    # store the buffer
    my_ma_config.update_from_dict({'store_buffer_in_checkpoints': True})

    # Add the MultiAgentCartPole env to our config and build our Algorithm.
    my_ma_config.environment(
        MultiAgentCartPole,
        env_config={
            "num_agents": 2,
        },
    )

    my_ma_algo = my_ma_config.build()
else:
    # to reload with different policies
    my_ma_algo = Algorithm.from_checkpoint(save_dir,
                                           policy_ids=set(policies),  # we still need to tell it to 'see' all policies
                                           policies_to_train=policies_to_train,  # train the new ones
                                           policy_mapping_fn=pol_map,
                                           # redefine polmap to give the correct new policies
                                           )

# if we are holding any agents untrainable, set the dummy weights to their weights
wait_dick = my_ma_algo.get_weights()
my_ma_algo.set_weights(
    {
        d_key: wait_dick[r_key]
        for d_key, (r_key, to_train) in zip(dummy_policies, policies_to_use) if not to_train
    }
)

# if we want to reset the buffers of various policies, delete them like this
for key in reset_replay:
    all_buffers = my_ma_algo.local_replay_buffer.replay_buffers
    if key in all_buffers:
        all_buffers.pop(key)

my_ma_algo.train()

for dumb_policy in dummy_policies:
    print('making sure',dumb_policy,'is untrained')
    for key in wait_dick[dumb_policy]:
        assert np.all(wait_dick[dumb_policy][key]==my_ma_algo.get_weights()[dumb_policy][key])


# we do not need to save dummy replay buffers, unsure why it even does this
for key in dummy_policies:
    all_buffers = my_ma_algo.local_replay_buffer.replay_buffers
    if key in all_buffers:
        all_buffers.pop(key)

ma_checkpoint_dir = my_ma_algo.save(save_dir).checkpoint.path

print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{ma_checkpoint_dir}'.\n"
    "Individual Policy checkpoints can be found in "
    f"'{os.path.join(ma_checkpoint_dir, 'policies')}'."
)

# Create a new Algorithm instance from the above checkpoint, just as you would for
# a single-agent setup:
# my_ma_algo_clone = Algorithm.from_checkpoint(ma_checkpoint_dir)
dick = pickle.load(open(os.path.join(save_dir, 'algorithm_state.pkl'), 'rb'))
for pol_to_check in policies:
    stuff = dick['local_replay_buffer']['replay_buffers']
    if pol_to_check in stuff:
        thing = stuff[pol_to_check]
        print(pol_to_check, ':')
        print('\tadded_count:', thing['added_count'])
        print('\tnum_entries:', thing['num_entries'])
    else:
        print(pol_to_check, 'untrained')
