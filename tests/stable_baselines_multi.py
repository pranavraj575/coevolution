from typing import Any, SupportsFloat

from gymnasium.core import ObsType, ActType
from pettingzoo.classic import rps_v2
from pettingzoo import ParallelEnv
from pettingzoo.butterfly import pistonball_v6
from stable_baselines3.dqn import DQN

from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.off_policy_algorithm import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
import numpy as np


class DumEnv(GymEnv):
    def __init__(self, action_space, obs_space, ):
        self.action_space = action_space
        self.observation_space = obs_space

    def reset(self, *, seed=None, options=None, ):
        raise NotImplementedError
        return self.observation_space.sample(), {}

    def step(self, action):
        raise NotImplementedError


class WorkerDQN(DQN):
    """
    meant to work inside a parallel DQN
    specifially broke the .learn() and .collect_rollout() methods
    now can iterate in a loop while broadcasting the actions taken to the parallel DQN
    """
    def __init__(self, policy, env):
        super().__init__(policy, env)

    def initialize_learn(self,
                         callback,
                         train_freq: TrainFreq,
                         ):
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        self.num_collected_steps, self.num_collected_episodes = 0, 0

        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise(self.env.num_envs)

        callback.on_rollout_start()

    def middle_of_rollout_select(self,
                                 obs,
                                 action_noise,
                                 learning_starts,
                                 ):
        # while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        if self.use_sde and self.sde_sample_freq > 0 and self.num_collected_steps%self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.actor.reset_noise(self.env.num_envs)

        self._last_obs = obs
        # Select action randomly or according to policy
        actions, buffer_actions = self._sample_action(learning_starts, action_noise, self.env.num_envs)
        return actions, buffer_actions

    def end_rollout_part(self,
                         new_obs,
                         reward,
                         termination,
                         truncation,
                         info,
                         callback,
                         replay_buffer,
                         buffer_actions,
                         action_noise,
                         learning_starts,
                         log_interval,
                         ):

        self.num_timesteps += self.env.num_envs
        self.num_collected_steps += 1

        # Give access to local variables
        callback.update_locals(locals())
        callback.on_step()
        # Only stop training if return value is False, not when it is None.
        # if not callback.on_step():
        #    return RolloutReturn(num_collected_steps*env.num_envs, num_collected_episodes, continue_training=False)
        dones = np.array([termination, truncation])
        # Retrieve reward and episode length if using Monitor wrapper
        self._update_info_buffer(infos, dones=dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones,
                               infos)  # type: ignore[arg-type]

        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        self._on_step()

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                self.num_collected_episodes += 1
                self._episode_num += 1

                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if self.env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

                # Log training infos
                if log_interval is not None and self._episode_num%log_interval == 0:
                    self._dump_logs()
        callback.on_rollout_end()

    def finished_with_rollout(self, callback):
        episode_timesteps = self.num_collected_steps*self.env.num_envs
        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

class ParallelDQN:
    def __init__(self, policy, parallel_env: ParallelEnv, workers=None):
        if workers is None:
            workers = dict()
        for agent in parallel_env.agents:
            if agent not in workers:
                dumenv = DumEnv(action_space=parallel_env.action_space(agent=agent),
                                obs_space=parallel_env.observation_space(agent=agent),
                                )
                workers[agent] = WorkerDQN(policy=policy, env=dumenv)
        self.workers = workers
        self.env = parallel_env

    def learn_episode(self,
                      callbacks=None,
                      log_interval=4,
                      ):
        if callbacks is None:
            callbacks = {agent: None for agent in self.workers}
        observations, infos = self.env.reset()
        local_callbacks = dict()
        for agent in self.workers:
            local_callbacks[agent] = self.workers[agent].initialize_learn(callback=callbacks[agent],
                                                                          train_freq=self.workers[agent].train_freq,
                                                                          )
        term = False
        while not term:
            actions = dict()
            buffer_actions = dict()
            for agent in self.workers:
                act, buff_act = self.workers[agent].middle_of_rollout_select(
                    observations[agent],
                    action_noise=self.workers[agent].action_noise,
                    learning_starts=self.workers[agent].learning_starts,
                )
                actions[agent] = act
                buffer_actions[agent] = act
            observations, rewards, terminations, truncations, infos = self.env.step(actions=actions)
            truncation = any([t for (_, t) in truncations.items()])
            termination = any([t for (_, t) in terminations.items()])
            for agent in self.workers:
                self.workers[agent].end_rollout_part(new_obs=observations[agent],
                                                     reward=rewards[agent],
                                                     termination=termination,
                                                     truncation=truncation,
                                                     info=infos[agent],
                                                     callback=local_callbacks[agent],
                                                     replay_buffer=self.workers[agent].replay_buffer,
                                                     buffer_actions=buffer_actions[agent],
                                                     action_noise=self.workers[agent].action_noise,
                                                     learning_starts=self.workers[agent].learning_starts,
                                                     log_interval=log_interval,
                                                     )
            term = truncation or termination
        for agent in self.workers:
            self.workers[agent].finished_with_rollout()


parallel_env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
observations, infos = parallel_env.reset(seed=42)

guy = 'piston_0'
test = DumEnv(action_space=parallel_env.action_space(guy), obs_space=parallel_env.observation_space(guy))
DQN(CnnPolicy, env=test, buffer_size=100)
quit()
print(observations[guy].shape)

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
    print([parallel_env.action_space(agent) for agent in parallel_env.agents])

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    print(rewards[guy])
    print(terminations[guy])
    print(truncations[guy])

    break
parallel_env.close()
