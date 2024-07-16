from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import ObsType, ActType
from gymnasium import spaces

from pettingzoo.classic import rps_v2
from pettingzoo import ParallelEnv
from pettingzoo.butterfly import pistonball_v6
from stable_baselines3.dqn import DQN

from stable_baselines3.dqn.policies import CnnPolicy, MlpPolicy
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.off_policy_algorithm import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
import numpy as np


class DumEnv(gymnasium.Env):
    def __init__(self, action_space, obs_space, ):
        self.action_space = action_space
        self.observation_space = obs_space

    def reset(self, *, seed=None, options=None, ):
        # need to implement this as setting up learning takes in an obs from here for some reason
        return self.observation_space.sample(), {}

    def step(self, action):
        raise NotImplementedError


def conform_shape(obs, obs_space):
    if obs_space.shape != obs.shape:
        if obs_space.shape[1:] == obs.shape[:2] and obs_space.shape[0] == obs.shape[2]:
            return np.transpose(obs, (2, 0, 1))
    return obs


class WorkerDQN(DQN):
    """
    meant to work inside a parallel DQN
    specifially broke the .learn() and .collect_rollout() methods
    now can iterate in a loop while broadcasting the actions taken to the parallel DQN
    """

    def __init__(self, policy, env, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)

    def initialize_learn(self,
                         callback,
                         total_timesteps,
                         train_freq: TrainFreq,
                         tb_log_name: str = "run",
                         reset_num_timesteps: bool = False,
                         progress_bar: bool = False,
                         ):

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        self.num_collected_steps, self.num_collected_episodes = 0, 0

        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise(self.env.num_envs)

        callback.on_rollout_start()
        return total_timesteps, callback

    def middle_of_rollout_select(self,
                                 obs,
                                 action_noise,
                                 learning_starts,
                                 ):
        # while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        if self.use_sde and self.sde_sample_freq > 0 and self.num_collected_steps%self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.actor.reset_noise(self.env.num_envs)

        actions, buffer_actions = self.get_action(obs=conform_shape(obs, self.observation_space),
                                                  learning_starts=learning_starts,
                                                  action_noise=action_noise,
                                                  )
        return actions, buffer_actions

    def get_action(self, obs, learning_starts=0, action_noise=None):

        self._last_obs = conform_shape(obs, self.observation_space)
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
        dones = np.array([termination or truncation])
        # Retrieve reward and episode length if using Monitor wrapper
        self._update_info_buffer(info, dones=dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        new_obs = conform_shape(new_obs, self.observation_space)
        self._store_transition(replay_buffer, buffer_actions, new_obs, reward, dones,
                               [info])  # type: ignore[arg-type]

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
    def __init__(self, policy, parallel_env: ParallelEnv, workers=None, worker_info=None, **worker_kwargs):
        """
        Args:
            policy: Type of policy to use for stableBaselines DQN
            parallel_env: Pettingzoo parallel env to use
            workers: dict of agentid -> worker
                trainable workers must have initialize_learn,middle_of_rollout_select,
                    get_action,end_rollout_part,finished_with_rollout
                untrainable must just have get_action
            worker_info: dict of agentid -> (worker info dict)
                worker info dict contains
                    train: bool (whether or not to treain worker)
            **worker_kwargs:
        """
        if workers is None:
            workers = dict()
        if worker_info is None:
            worker_info = dict()
        for agent in parallel_env.agents:
            if agent not in workers:
                dumenv = DumEnv(action_space=parallel_env.action_space(agent=agent),
                                obs_space=parallel_env.observation_space(agent=agent),
                                )
                workers[agent] = WorkerDQN(policy=policy, env=dumenv, **worker_kwargs)
            if agent not in worker_info:
                worker_info[agent] = {
                    'train': True
                }
        self.workers = workers
        self.worker_info = worker_info
        self.env = parallel_env

    def learn(self,
              total_timesteps,
              callbacks=None,
              log_interval=4,
              ):
        while total_timesteps > 0:
            timesteps = self.learn_episode(total_timesteps=total_timesteps,
                                           callbacks=callbacks,
                                           log_interval=log_interval,
                                           )
            total_timesteps -= timesteps

    def _get_worker_iter(self, trainable):
        """
        Args:
            trainable: if true, returns trainable workers
                else, untrainable workers
        Returns: iterable of trainable or untrainable workers
        """
        for agent in self.workers:
            is_trainable = self.worker_info[agent].get('train', True)
            if is_trainable == trainable:  # either both true or both false
                yield agent

    def get_trainable_workers(self):
        return self._get_worker_iter(trainable=True)

    def get_untrainable_workers(self):
        return self._get_worker_iter(trainable=False)

    def learn_episode(self,
                      total_timesteps,
                      callbacks=None,
                      log_interval=4,
                      ):
        if callbacks is None:
            callbacks = {agent: None for agent in self.workers}
        observations, infos = self.env.reset()
        local_callbacks = dict()
        for agent in self.get_trainable_workers():
            ag_total_timesteps, ag_callback = self.workers[agent].initialize_learn(
                total_timesteps=total_timesteps,
                callback=callbacks[agent],
                train_freq=self.workers[agent].train_freq,
            )
            local_callbacks[agent] = ag_callback
        term = False
        while not term:
            actions = dict()
            buffer_actions = dict()
            for agent in self.get_trainable_workers():
                act, buff_act = self.workers[agent].middle_of_rollout_select(
                    observations[agent],
                    action_noise=self.workers[agent].action_noise,
                    learning_starts=self.workers[agent].learning_starts,
                )
                if isinstance(self.workers[agent].action_space, spaces.Discrete) and not isinstance(act, int):
                    actions[agent] = act.reshape(1)[0]
                else:
                    actions[agent] = act
                buffer_actions[agent] = buff_act
            for agent in self.get_untrainable_workers():
                act = self.workers[agent].get_action(
                    obs=observations[agent]
                )
                actions[agent] = act
            observations, rewards, terminations, truncations, infos = self.env.step(actions=actions)
            truncation = any([t for (_, t) in truncations.items()])
            termination = any([t for (_, t) in terminations.items()])
            for agent in self.get_trainable_workers():
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
        num_collected_steps = 0
        for agent in self.get_trainable_workers():
            self.workers[agent].finished_with_rollout(callback=local_callbacks[agent], )
            num_collected_steps = self.workers[agent].num_collected_steps

        return num_collected_steps


env = rps_v2.parallel_env(render_mode="human")
observations, infos = env.reset()

class always_0:
    def get_action(self,*args,**kwargs):
        return 0

thingy = ParallelDQN(policy=MlpPolicy,
                     parallel_env=env,
                     buffer_size=100,
                     worker_info={'player_1':{'train':False}},
                     workers={'player_1':always_0()},
                     learning_starts=10
                     )
thingy.learn(total_timesteps=100)


quit()

parallel_env = pistonball_v6.parallel_env(render_mode="human", continuous=False, n_pistons=6)
observations, infos = parallel_env.reset(seed=42)

thingy = ParallelDQN(policy=CnnPolicy, parallel_env=parallel_env, buffer_size=1000)
thingy.learn(total_timesteps=1000)

quit()

guy = 'piston_0'
# test = DumEnv(action_space=parallel_env.action_space(guy), obs_space=parallel_env.observation_space(guy))
# DQN(CnnPolicy, env=test, buffer_size=100)
print(observations[guy].shape)

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}
    # print([parallel_env.action_space(agent) for agent in parallel_env.agents])
    print(actions)

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    print(rewards[guy])
    print(terminations[guy])
    print(truncations[guy])

parallel_env.close()
