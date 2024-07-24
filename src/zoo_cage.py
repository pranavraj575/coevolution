import os, pickle, shutil
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import BaseBuffer, DictReplayBuffer, ReplayBuffer, RolloutBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl


class ZooCage:
    def __init__(self, zoo_dir, overwrite_zoo=True):
        self.zoo_dir = zoo_dir
        if not os.path.exists(self.zoo_dir):
            os.makedirs(self.zoo_dir)
        elif overwrite_zoo:
            self.kill_zoo()
            os.makedirs(self.zoo_dir)
        self.active_workers = dict()
        self.saved_workers = set(os.listdir(self.zoo_dir))

    def activate_worker(self, agent_key, worker_key, WorkerClass=None, load_buffer=True):
        """
        loads worker and adds to active workers
        Args:
            agent_key: key to save worker as in dictionary (usually same as pettingzoo enviornment agent id)
            worker_key: folder to grab worker from
            WorkerClass: class to use to load worker (if none, tries to load from class.pkl)
            load_buffer: whether to load replay buffer, if available
        """
        self.active_workers[agent_key] = self.load_worker(worker_key=worker_key,
                                                          WorkerClass=WorkerClass,
                                                          load_buffer=load_buffer,
                                                          )

    def load_worker(self, worker_key: str, WorkerClass=None, load_buffer=True):
        """
        loads worker from specified folder
        Args:
            worker_key: folder to grab worker from
            WorkerClass: class to use to load worker (if none, tries to load from class.pkl)
            load_buffer: whether to load replay buffer, if available
        Returns: loaded worker (SB3 algorithm)
        """
        full_dir = os.path.join(self.zoo_dir, worker_key)
        if WorkerClass is None:
            f = open(os.path.join(full_dir, 'class.pkl'), 'rb')
            WorkerClass = pickle.load(f)
            f.close()
        worker = WorkerClass.load(os.path.join(full_dir, 'worker'))
        if load_buffer:
            if isinstance(worker, OffPolicyAlgorithm):
                buff_file = os.path.join(full_dir, 'replay_buffer.pkl')
                if os.path.exists(buff_file):
                    worker.load_replay_buffer(buff_file)
                else:
                    print('buffer file not found:', buff_file)
                    print('change saving settings perhaps, unless this was intended')
            elif isinstance(worker, OnPolicyAlgorithm):
                worker.rollout_buffer = load_from_pkl(os.path.join(full_dir, 'rollout_buffer.pkl'))
                worker.rollout_buffer.device = worker.device
            else:
                print('WHAT ALGORITHM IS THIS', worker)
        return worker

    def save_active_worker(self, agent_key, worker_key: str, save_buffer=True, save_class=True):
        """
        saves worker to folder named worker_key
        Args:
            agent_key: key of agent in active_worker dict (usually something like 'player 0' in environment)
            worker_key: worker to save, folder to save to
            save_buffer: whether to save replay buffer, if available
            save_class: whether to save class of algorithm (suprised this works)
        """
        self.overwrite_worker(worker=self.active_workers[agent_key],
                              worker_key=worker_key,
                              save_buffer=save_buffer,
                              save_class=save_class,
                              )

    def update_worker_buffer(self, agent_key, worker_key: str):
        """
        takes active agent's buffer and stores it into buffer of worker_key
            assumes buffer inherits BaseBuffer
        Args:
            agent_key:
            worker_key: folder to load from/save to
        Returns:

        """
        raise NotImplementedError

    def overwrite_worker(self, worker, worker_key: str, save_buffer=True, save_class=True):
        """
        saves worker to folder named worker_key
        Args:
            worker: worker to save
            worker_key: folder to save to
            save_buffer: whether to save replay buffer, if available
            save_class: whether to save class of algorithm (suprised this works)
        """
        full_dir = os.path.join(self.zoo_dir, worker_key)

        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        # worker inherits save method from stable baselines
        worker.save(os.path.join(full_dir, 'worker'))
        if save_buffer:
            if isinstance(worker, OffPolicyAlgorithm):
                # worker has a replay buffer that should also probably be saved
                worker.save_replay_buffer(os.path.join(full_dir, 'replay_buffer.pkl'))
            elif isinstance(worker, OnPolicyAlgorithm):
                save_to_pkl(os.path.join(full_dir, 'rollout_buffer.pkl'), worker.rollout_buffer)
            else:
                print('WHAT ALGORITHM IS THIS', worker)
        if save_class:
            cls = type(worker)
            f = open(os.path.join(full_dir, 'class.pkl'), 'wb')
            pickle.dump(cls, f)
            f.close()
        self.saved_workers.add(worker_key)

    def save_zoo(self, save_dir):
        shutil.rmtree(save_dir)
        shutil.copytree(self.zoo_dir, save_dir)

    def load_zoo(self, save_dir):
        self.kill_zoo()
        shutil.copytree(save_dir, self.zoo_dir)
        self.saved_workers = set(os.listdir(self.zoo_dir))

    def kill_zoo(self):
        shutil.rmtree(self.zoo_dir)


if __name__ == '__main__':
    import sys
    from parallel_algs.ppo.PPO import WorkerPPO
    from stable_baselines3.ppo import MlpPolicy

    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    zoo = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_cage'))
    for i in range(100):
        zoo.overwrite_worker(worker=WorkerPPO(policy=MlpPolicy, env='CartPole-v0'),
                             worker_key=str(i),
                             )
    save_dir = os.path.join(DIR, 'data', 'zoo_cage_save')
    zoo.save_zoo(save_dir=save_dir)
    zoo2 = ZooCage(zoo_dir=os.path.join(DIR, 'data', 'zoo_cage2'))
    zoo2.load_zoo(save_dir=save_dir)
    print(len(zoo2.saved_workers))
    zoo.kill_zoo()
    zoo2.kill_zoo()
    shutil.rmtree(save_dir)
