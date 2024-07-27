import torch
import os, sys, shutil, pickle


class LangReplayBuffer:
    def reset_storage(self):
        """
        resets internal buffer
        """
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.push(item)

    def push(self, item):
        raise NotImplementedError

    def sample_one(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def sample(self, batch):
        for _ in range(batch):
            yield self.sample_one()

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ReplayBufferDiskStorage(LangReplayBuffer):
    def __init__(self, storage_dir, capacity=1e6, device=None):
        self.idx = 0
        self.size = 0
        self.storage_dir = storage_dir
        self.capacity = capacity
        self.reset_storage()
        self.device = device

    def delete(self):
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    def reset_storage(self):
        self.delete()
        os.makedirs(self.storage_dir)
        self.size = 0
        self.idx = 0

    def save_place(self):
        """
        saves idx and size to files as well
        """
        pickle.dump(
            {
                'size': self.size,
                'idx': self.idx,
            },
            open(self._get_file('info'), 'wb')
        )

    def load_place(self, force=False):
        info_file = self._get_file(name='info')
        if os.path.exists(info_file):
            dic = pickle.load(open(info_file, 'rb'))
            self.size = dic['size']
            self.idx = dic['idx']
        else:
            if force:
                print('failed to load file:', info_file)
                print('resetting storage')
                self.reset_storage()
            else:
                raise Exception('failed to load file: ' + info_file)

    def _get_file(self, name):
        return os.path.join(self.storage_dir, str(name) + '.pkl')

    def push(self, item):
        pickle.dump(item, open(self._get_file(self.idx), 'wb'))

        self.size = max(self.idx + 1, self.size)
        self.idx = int((self.idx + 1)%self.capacity)

        self.save_place()

    def _grab_item_by_idx(self, idx, change_device=True):
        item = pickle.load(open(self._get_file(name=idx), 'rb'))
        return self._convert_device(item=item, change_device=change_device)

    def _convert_device(self, item, change_device):
        if change_device:
            if type(item) == tuple:
                item = tuple(self._convert_device(t, change_device=change_device)
                             for t in item)
            elif torch.is_tensor(item):
                item = item.to(self.device)
        return item

    def sample_one(self):
        return self[torch.randint(0, self.size, (1,))]

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError
        return self._grab_item_by_idx(idx=int((self.idx + item)%self.size))

    def __len__(self):
        return self.size


if __name__ == '__main__':
    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    test = ReplayBufferDiskStorage(capacity=3, storage_dir=os.path.join(DIR, 'data', 'buffers', 'replay_buffer_test'))
    test.extend('help')
    print(list(test.sample(3)))
