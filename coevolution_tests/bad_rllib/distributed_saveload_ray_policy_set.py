import pickle
import os, sys
import time

import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import MultiAgentReplayBuffer
from ray.rllib.policy.policy import Policy
import copy
# Use our example multi-agent CartPole environment to train in.
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole

import logging

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step, multi_gpu_train_one_step
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_TARGET_UPDATES,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
from ray.rllib.utils.typing import RLModuleSpec, SampleBatchType
from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights

logger = logging.getLogger(__name__)


class GlobalDQN(DQN):
    """
    inherits dqn, on training step it does not sample the enviornment, it instead just trains on replay buffer
    """

    def _training_step_old_and_hybrid_api_stack(self) -> ResultDict:
        """Training step for the old and hybrid training stacks.

        More specifically this training step relies on `RolloutWorker`.
        """
        train_results = {}

        # We alternate between storing new samples and sampling and training
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)

        """
        for _ in range(store_weight):
            # Sample (MultiAgentBatch) from workers.
            with self._timers[SAMPLE_TIMER]:
                new_sample_batch: SampleBatchType = synchronous_parallel_sample(
                    worker_set=self.workers, concat=True
                )

            # Update counters
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()

            # Store new samples in replay buffer.
            self.local_replay_buffer.add(new_sample_batch)

        """
        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }

        # Update target network every `target_network_update_freq` sample steps.
        cur_ts = self._counters[
            (
                NUM_AGENT_STEPS_SAMPLED
                if self.config.count_steps_by == "agent_steps"
                else NUM_ENV_STEPS_SAMPLED
            )
        ]
        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(sample_and_train_weight):
                # Sample training batch (MultiAgentBatch) from replay buffer.
                train_batch = sample_min_n_steps_from_buffer(
                    self.local_replay_buffer,
                    self.config.train_batch_size,
                    count_by_agent_steps=self.config.count_steps_by == "agent_steps",
                )

                # Postprocess batch before we learn on it
                post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)

                # Learn on training batch.
                # Use simple optimizer (only for multi-agent or tf-eager; all other
                # cases should use the multi-GPU optimizer, even if only using 1 GPU)
                if self.config.get("simple_optimizer") is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)
                # Update replay buffer priorities.
                update_priorities_in_replay_buffer(
                    self.local_replay_buffer,
                    self.config,
                    train_batch,
                    train_results,
                )

                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config.target_network_update_freq:
                    to_updat = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(
                        lambda p, pid, to_update=to_updat: (
                                pid in to_update and p.update_target()
                        )
                    )
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

                # Update weights and global_vars - after learning on the local worker -
                # on all remote workers.

                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

        # Return all collected metrics for the iteration.
        return train_results


# Set up a multi-agent Algorithm, training two policies independently.
dummy_policies = ['dummy1', 'dummy2']
saved_policies = ['pol' + str(i) for i in range(10)]

policies_to_use = [('pol8', True), ('pol9', True)]
# which to put in for agents 0 and 1, and whether to train them
# if False, the policy will be used, it will be fixed, and the replay buffer will not be updated

#policies_to_update=['pol3']
policies_to_update=saved_policies
# call update step on these policiees


reset_replay = []
# reset_replay = ['pol0']
reset_all = False


def pol_map(agent_id, episode, worker, **kw):
    # policies_to_use[id] is the correct policy
    # however, if we are holding any fixed, we map it to the appropriate dummy
    return dummy_policies[agent_id]

    thing, to_train = policies_to_use[agent_id]
    if to_train:
        return thing
    else:
        return dummy_policies[agent_id]


def global_pol_map(agent_id, episode, worker, **kw):
    # policies_to_use[id] is the correct policy
    # however, if we are holding any fixed, we map it to the appropriate dummy
    return saved_policies[agent_id]


DIR = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))

save_dir = os.path.join(DIR, '..', 'data', 'TEST')

if not os.path.exists(save_dir) or reset_all:
    global_config = DQNConfig(algo_class=GlobalDQN).multi_agent(
        # Which policies should RLlib create and train?
        # initialize all policies, only train the important ones
        policies=set(saved_policies),
        # Let RLlib know, which agents in the environment (we'll have "agent1"
        # and "agent2") map to which policies.
        policy_mapping_fn=global_pol_map,
        policies_to_train=policies_to_update,
    )
    global_config.training(
        replay_buffer_config={'type': MultiAgentReplayBuffer}
    )

    # store the buffer
    global_config.update_from_dict({'store_buffer_in_checkpoints': True})

    # Add the MultiAgentCartPole env to our config and build our Algorithm.
    global_config.environment(
        MultiAgentCartPole,
        env_config={
            "num_agents": len(saved_policies),
        },
    )

    global_algo = global_config.build()
else:
    # to reload with different policies
    global_algo = Algorithm.from_checkpoint(save_dir,
                                            policy_ids=set(saved_policies),
                                            # we still need to tell it to 'see' all policies
                                            policies_to_train=policies_to_update,  # train the new ones
                                            policy_mapping_fn=pol_map,
                                            # redefine polmap to give the correct new policies
                                            )

"""
PARALLIZABLE LOCAL TRAINING

after loading the weights and other params, sets these little dudes off to collect experience
"""
# NORMAL DQN, except no weight updates to any agents
local_config = DQNConfig().multi_agent(
    # Which policies should RLlib create and train?
    # initialize all policies, only train the important ones
    policies=set(dummy_policies),  # workers only see dummy policies
    # Let RLlib know, which agents in the environment (we'll have "agent1"
    # and "agent2") map to which policies.
    policy_mapping_fn=pol_map,
    policies_to_train=[],  # do not train in the worker, just collect experience

)
local_config.training(
    replay_buffer_config={'type': MultiAgentReplayBuffer}
)

# store the buffer
local_config.update_from_dict({'store_buffer_in_checkpoints': True})

# Add the MultiAgentCartPole env to our config and build our Algorithm.
local_config.environment(
    MultiAgentCartPole,
    env_config={
        "num_agents": 2,
    },
)

local_algo = local_config.build()


# set all local keys to correct global keys
global_waits = copy.deepcopy(global_algo.get_weights())
local_algo.set_weights(
    {
        local_key: global_waits[global_key]
        for local_key, (global_key, to_train) in zip(dummy_policies, policies_to_use)
    }
)

all_buffers = global_algo.local_replay_buffer.replay_buffers
# if we want to reset the buffers of various policies, delete them like this
for key in reset_replay:
    if key in all_buffers:
        all_buffers.pop(key)

# mess up the buffers to see if it is actually training from them
# for key in all_buffers:
#    all_buffers[key]._storage[0]['obs'] = None
# for key2 in all_buffers[key]._storage[0]:
#    print(key2, all_buffers[key]._storage[0][key2])

local_waits = copy.deepcopy(local_algo.get_weights())
local_algo.train()
local_algo.train()
print(local_algo.config.epsilon)

for dumb_policy in dummy_policies:
    print('making sure', dumb_policy, 'is untrained')
    for key in local_waits[dumb_policy]:
        assert np.all(local_waits[dumb_policy][key] == local_algo.get_weights()[dumb_policy][key])

# TODO: save dummy replay buffers to correct buffer here
local_buffers = local_algo.local_replay_buffer.replay_buffers
global_buffers = global_algo.local_replay_buffer.replay_buffers

for local_key, (real_key, to_train) in zip(dummy_policies, policies_to_use):
    if to_train:
        # print(dir(local_buffers[local_key]))
        # print(local_buffers[local_key]._storage[0])
        for batch in local_buffers[local_key]._storage:
            global_buffers[real_key].add(batch)
for key in local_algo._counters:
    global_algo._counters[key] += local_algo._counters[key]
print(global_algo._counters)

# TODO: global weight update
print('updating global algo')
global_algo.training_step()

ma_checkpoint_dir = global_algo.save(save_dir).checkpoint.path

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
for pol_to_check in saved_policies:
    stuff = dick['local_replay_buffer']['replay_buffers']
    if pol_to_check in stuff:
        thing = stuff[pol_to_check]
        print(pol_to_check, ':')
        print('\tadded_count:', thing['added_count'])
        print('\tnum_entries:', thing['num_entries'])

        changed = [(key, np.any(global_waits[pol_to_check][key] != global_algo.get_weights()[pol_to_check][key]))
                   for key in global_waits[pol_to_check]]
        print('weights changed', any(t for _,t in changed))
    else:
        print(pol_to_check, 'untrained')
