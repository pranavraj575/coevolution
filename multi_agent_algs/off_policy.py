import numpy as np

from stable_baselines3.dqn import DQN
from stable_baselines3.common.type_aliases import TrainFreq
from common import conform_shape


class OffPolicy:
    def init_learn(self,
                   callback,
                   total_timesteps,
                   train_freq: TrainFreq = None,
                   tb_log_name: str = "run",
                   reset_num_timesteps: bool = False,
                   progress_bar: bool = False,
                   log_interval=4,
                   ):
        if train_freq is None:
            train_freq = self.train_freq

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

        init_learn_info = {'callback': callback,
                           'log_interval': log_interval,
                           }
        return init_learn_info

    def init_rollout(self,
                     init_learn_info
                     ):
        train_freq = self.train_freq
        callback = init_learn_info.get('callback')

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        self.num_collected_steps, self.num_collected_episodes = 0, 0

        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise(self.env.num_envs)

        callback.on_rollout_start()
        return None

    def rollout_1(self,
                  init_learn_info,
                  init_rollout_info,
                  ):
        # while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        if self.use_sde and self.sde_sample_freq > 0 and self.num_collected_steps%self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.actor.reset_noise(self.env.num_envs)
        return None

    def rollout_2(self,
                  obs,
                  init_learn_info,
                  init_rollout_info,
                  rollout_1_info,
                  ):
        action_noise = self.action_noise
        learning_starts = self.learning_starts
        actions, rollout_2_info = self.get_action(obs=conform_shape(obs, self.observation_space),
                                                  learning_starts=learning_starts,
                                                  action_noise=action_noise,
                                                  )
        return actions, rollout_2_info

    def rollout_3(self,
                  action,
                  new_obs,
                  reward,
                  termination,
                  truncation,
                  info,
                  init_learn_info,
                  init_rollout_info,
                  rollout_1_info,
                  rollout_2_info,
                  replay_buffer=None,
                  ):
        buffer_action = rollout_2_info
        callback = init_learn_info.get('callback')
        log_interval = init_learn_info.get('log_interval', None)
        action_noise = self.action_noise
        if replay_buffer is None:
            replay_buffer = self.replay_buffer

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
        self._store_transition(replay_buffer, buffer_action, new_obs, reward, dones,
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
        # if not done, then return true for continuing rollout
        # else, return false for ending rollout
        return {
            'continue_rollout': not np.any(dones),
            'steps_so_far': self.num_collected_steps,
        }

    def get_action(self, obs, learning_starts=0, action_noise=None):

        self._last_obs = conform_shape(obs, self.observation_space)
        # Select action randomly or according to policy
        actions, buffer_actions = self._sample_action(learning_starts, action_noise, self.env.num_envs)
        return actions, buffer_actions

    def end_rollout(self,
                    init_learn_info,
                    init_rollout_info,
                    ):
        callback = init_learn_info.get('callback')
        callback.on_rollout_end()
        return {'num_collected_steps': self.num_collected_steps}

    def finish_learn(self, init_learn_info, end_rollout_info):
        callback = init_learn_info.get('callback')
        episode_timesteps = self.num_collected_steps*self.env.num_envs
        if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()
