import os, shutil
import dill as pickle

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl


def overwrite_other(other,
                    save_dir,
                    other_info=None,
                    ):
    """
    pickles another object and saves it into key
    Args:
        other: some object
        other_key: folder to save into
        other_info: info to save
    Returns:
    """

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'other.pkl'), 'wb')
    pickle.dump(other, f)
    f.close()

    overwrite_info(info=other_info,
                   save_path=os.path.join(save_dir, 'info.pkl'),
                   )


def load_other(save_dir):
    f = open(os.path.join(save_dir, 'other.pkl'), 'rb')
    other = pickle.load(f)
    f.close()
    return other, load_info(save_path=os.path.join(save_dir, 'info.pkl'))


def other_exists(save_dir):
    return (os.path.exists(os.path.join(save_dir, 'info.pkl')) and
            os.path.exists(os.path.join(save_dir, 'other.pkl')))


def overwrite_info(info,
                   save_path,
                   ):
    if info is None:
        info = dict()
    f = open(save_path, 'wb')
    pickle.dump(info, f)
    f.close()


def load_info(save_path):
    if os.path.exists(save_path):
        f = open(save_path, 'rb')
        info = pickle.load(f)
        f.close()
        return info
    else:
        return dict()


def overwrite_worker(worker,
                     worker_info,
                     save_dir,
                     save_buffer=True,
                     save_class=True,
                     ):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # save info
    overwrite_info(info=worker_info,
                   save_path=os.path.join(save_dir, 'info.pkl'),
                   )

    # save class
    if save_class:
        cls = type(worker)
        f = open(os.path.join(save_dir, 'class.pkl'), 'wb')
        pickle.dump(cls, f)
        f.close()

    # save buffer
    if save_buffer:
        if isinstance(worker, OffPolicyAlgorithm):
            worker.save_replay_buffer(os.path.join(save_dir, 'replay_buffer.pkl'))
        elif isinstance(worker, OnPolicyAlgorithm):
            save_to_pkl(os.path.join(save_dir, 'rollout_buffer.pkl'), worker.rollout_buffer)
        else:
            raise Exception('type not valid', type(worker))
    # save worker
    worker.save(os.path.join(save_dir, 'worker.zip'))


def load_worker(save_dir,
                WorkerClass=None,
                load_buffer=True,
                ):
    # load info
    filename = os.path.join(save_dir, 'info.pkl')
    if os.path.exists(filename):
        f = open(filename, 'rb')
        worker_info = pickle.load(f)
        f.close()
    else:
        worker_info = {}

    # load class
    if WorkerClass is None:
        f = open(os.path.join(save_dir, 'class.pkl'), 'rb')
        WorkerClass = pickle.load(f)
        f.close()

    # load worker
    worker = WorkerClass.load(os.path.join(save_dir, 'worker.zip'))

    # load buffer
    if load_buffer:
        if isinstance(worker, OffPolicyAlgorithm):
            buff_file = os.path.join(save_dir, 'replay_buffer.pkl')
            if os.path.exists(buff_file):
                worker.load_replay_buffer(buff_file)
        elif isinstance(worker, OnPolicyAlgorithm):
            buff_file = os.path.join(save_dir, 'rollout_buffer.pkl')
            if os.path.exists(buff_file):
                worker.rollout_buffer = load_from_pkl(buff_file)
                worker.rollout_buffer.device = worker.device
        else:
            print('WHAT ALGORITHM IS THIS', worker)

    return worker, worker_info


def worker_exists(save_dir):
    return (os.path.exists(os.path.join(save_dir, 'info.pkl')) and
            os.path.exists(os.path.join(save_dir, 'worker.zip')))
