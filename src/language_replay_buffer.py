import torch, os, sys, shutil
import dill as pickle


class LangReplayBuffer:
    storage_dir = None

    def set_storage_dir(self, storage_dir, reset=False):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        if reset:
            self.reset_storage()

    def reset_storage(self):
        """
        resets internal buffer
        """
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.push(item)

    def push(self, item):
        """
        pushes an item into replay buffer
        Args:
            item: item
        Returns: item that is displaced, or None if no such item
        """
        raise NotImplementedError

    def sample_one(self):
        raise NotImplementedError

    def clear(self):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def sample(self, batch, **kwargs):
        for _ in range(batch):
            yield self.sample_one()

    # def __getitem__(self, item):
    #    raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ReplayBufferDiskStorage(LangReplayBuffer):
    def __init__(self,
                 storage_dir=None,
                 capacity=1e6,
                 device=None,
                 ):
        self.idx = 0
        self.size = 0
        self.capacity = capacity
        self.device = device
        if storage_dir is not None:
            self.set_storage_dir(storage_dir=storage_dir)

    def clear(self):
        super().clear()
        if self.storage_dir is not None:
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        shutil.copytree(src=self.storage_dir, dst=save_dir)

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.clear()
        shutil.copytree(src=save_dir, dst=self.storage_dir)

    def reset_storage(self):
        self.clear()
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
        if self.size == self.capacity:
            disp = self.__getitem__(self.idx)
        else:
            disp = None
        pickle.dump(item, open(self._get_file(self.idx), 'wb'))

        self.size = max(self.idx + 1, self.size)
        self.idx = int((self.idx + 1)%self.capacity)

        self.save_place()
        return disp

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


class BinnedReplayBufferDiskStorage(LangReplayBuffer):
    """
    creates multiple disk replay buffers, each representing a 'bin' of data
    items must begin with a scalar that represents how much it should show up in the data
    """

    def __init__(self,
                 storage_dir=None,
                 bounds=None,
                 capacity=1e6,
                 device=None,
                 overwrite_existing=True,
                 ):
        """
        Args:
            bounds: sorted list of numbers to divide the input into bins based on the scalar associated with it
                each bin keeps track of the average value
                upon sampling, an extra parameter can be added to determine with what
                    frequency elements from each bin appear

                values <=bounds[0] or >bounds[1] are ignored
                by default, uses [1/2, 1], corresponding to one bin of range (1/2,1]
                    This is intended for a 2 player game where we want to capture outcomes that are better than ties
                        Usually, in an n-player game, the bounds should range [1/n,...,1] to capture outcomes that are
                            better than ties
            overwrite_existing: whether to overwrite existing storage

        """
        super().__init__()
        if bounds is None:
            bounds = [1/2, 1]
        self.bins = [ReplayBufferDiskStorage(storage_dir=None,
                                             capacity=capacity,
                                             device=device)
                     for i in range(len(bounds) - 1)
                     ]
        self.info = {
            'bounds': tuple(bounds),
            'avgs': torch.zeros(len(bounds) - 1),
            'size': 0,
        }
        if storage_dir is not None:
            self.set_storage_dir(storage_dir=storage_dir, reset=overwrite_existing)

    def set_storage_dir(self, storage_dir, reset=False):
        super().set_storage_dir(storage_dir=storage_dir, reset=reset)
        for i, biin in enumerate(self.bins):
            biin.set_storage_dir(storage_dir=os.path.join(storage_dir, 'bin_' + str(i)),
                                 reset=reset)

    def reset_storage(self):
        self.clear()
        os.makedirs(self.storage_dir)
        self.idx = 0

    def set_size(self, size):
        self.info['size'] = size

    @property
    def size(self):
        return self.info['size']

    @property
    def bounds(self):
        return self.info['bounds']

    @property
    def avgs(self):
        return self.info['avgs']

    def bin_search(self, value, possible=None):
        """
        binary bin search
        Args:
            value: value to put into bin
            possible: possible bindices (if None, searches all)
        Returns:
            i such that self.bounds[i] < value<= self.bounds[i+1]
            or None if not possible
        """
        if possible is None:
            possible = (0, len(self.bounds) - 1)
        i, j = possible
        if i + 1 == j:
            if self.bounds[i] < value and value <= self.bounds[i + 1]:
                return i
            else:
                return None
        mid = (i + j)//2
        # i+1 <= mid <= j-1
        if self.bounds[mid] < value:
            return self.bin_search(value, (mid, j))
        if value < self.bounds[mid]:
            return self.bin_search(value, (i, mid))
        if value == self.bounds[mid]:
            # 0 <= i <= mid-1
            return mid - 1

    def push(self, item):
        scalar = item[0]
        biin = self.bin_search(scalar)
        if biin is not None:
            disp = self.bins[biin].push(item)
            if disp is None:
                rem = 0
                self.set_size(self.size + 1)
            else:
                rem = disp[0]

            self.avgs[biin] = (self.avgs[biin]*(len(self.bins[biin]) - 1) + scalar - rem)/len(self.bins[biin])

    def set_weights(self, values_to_weights=None):
        """
        sets weights of each  bin according to the average value of its elements
        Args:
        values_to_weights: a function (tensor -> tensor) ([0,1] -> R+), weights to give a bin with a particular value
            if None, uses weights of len(bin) for each bin (this corresponds to uniformly sampling an element)
        """
        if values_to_weights == None:
            self.weights = self.bin_lens
        else:
            self.weights = values_to_weights(self.avgs)

    def sample_one(self):
        biin = torch.multinomial(self.weights, 1)
        return self.bins[biin].sample_one()

    def sample(self, batch, values_to_weights=None, **kwargs):
        """
        samples a bin according to the average value of its elements, then samples elements from the bin
        Args:
            batch: number of elements to sample
            values_to_weights: a function (tensor -> tensor) ([0,1] -> R+), weights to give a bin with a particular value
                if None, uses weights of len(bin) for each bin (this corresponds to uniformly sampling an element)
            **kwargs:
        Returns:
            Iterable of samples
        """
        self.set_weights(values_to_weights=values_to_weights)
        for item in super().sample(batch=batch, **kwargs):
            yield item

    def clear(self):
        super().clear()
        for guy in self.bins:
            guy.clear()
        if self.storage_dir is not None:
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        f = open(os.path.join(save_dir, 'info.pkl'), 'wb')
        pickle.dump(self.info, f)
        f.close()

        for i, guy in enumerate(self.bins):
            guy.save(save_dir=os.path.join(save_dir, 'bin_' + str(i)))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        f = open(os.path.join(save_dir, 'info.pkl'), 'rb')
        self.info.update(pickle.load(f))
        f.close()

        for i, guy in enumerate(self.bins):
            guy.load(save_dir=os.path.join(save_dir, 'bin_' + str(i)))

    def __len__(self):
        return self.size

    @property
    def bin_lens(self):
        return torch.tensor([len(bin) for bin in self.bins],
                            dtype=torch.float,
                            )


if __name__ == '__main__':
    DIR = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), sys.argv[0])))
    test = ReplayBufferDiskStorage(capacity=3, storage_dir=os.path.join(DIR, 'data', 'buffers', 'replay_buffer_test'))
    test.extend('help')
    print(list(test.sample(3)))
