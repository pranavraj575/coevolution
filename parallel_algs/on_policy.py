from gymnasium import spaces

from pettingzoo import ParallelEnv

class OnPolicy:
    def init_learn(self,
                   callback,
                   total_timesteps,
                   tb_log_name: str = "run",
                   reset_num_timesteps: bool = False,
                   progress_bar: bool = False,
                   log_interval=4,
                   ):
        self.iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"

        init_learn_info = {'callback': callback,
                           'log_interval': log_interval,
                           'total_timesteps': total_timesteps,
                           }
        return init_learn_info
