from pathos.multiprocessing import Pool

import time
import os
from matplotlib import pyplot as plt
from multiprocessing.pool import ApplyResult

def f(x):
    i = 0
    tim = time.time()
    for _ in range(x):
        i += 1
    return {'tim':time.time() - tim}


add_ops = 3169000


def get_async_list(processes, fn, inputs):
    pool = Pool(processes=processes, )
    return pool, [pool.apply_async(fn, (inp,)) for inp in inputs]


def mutate(item, pool, fn, inp):
    if not isinstance(item, dict):
        if item.ready():
            return item.get()
        else:
            return pool.apply_async(fn, (inp,))
    else:
        return item


def impatience_reset(processes, inputs, fn, pool=None, res=None):
    if pool is not None:
        try:
            pool.close()
            pool.terminate()
        except:
            print('err')
    pool = Pool(processes=processes, )
    done = False

    if res is None:
        res = [pool.apply_async(fn, (inp,)) for inp in inputs]
    else:
        res = [mutate(item, pool, fn, inp) for item, inp in zip(res, inputs)]
        if all([isinstance(item, dict) for item in res]):
            done = True
        else:
            print(res[-1].terminate())
    return pool, res, done


inputs = [add_ops for _ in range(500)]
pool = None
res = None
done = False
old = 0

while not done:
    pool, res, done = impatience_reset(processes=8, inputs=inputs, fn=f, pool=pool, res=res)
    time.sleep(.5)
    thing = sum([True if type(item) == dict else item.ready() for item in res])
    print('total', thing, 'change', thing - old)
    old = thing

print('here')
quit()

max_cores = len(os.sched_getaffinity(0))


def f(x):
    i = 0
    tim = time.time()
    for _ in range(x):
        i += 1
    return time.time() - tim


add_ops = 316900
runtimes = []
seq_runtimes = []
for processes in range(1, max_cores*3):
    with Pool(processes=processes) as pool:
        # print "[0, 1, 4,..., 81]"
        tim = time.time()
        print(pool.map(f, [add_ops for _ in range(processes)]))
        runtimes.append(time.time() - tim)

    with Pool(processes=1) as pool:
        # print "[0, 1, 4,..., 81]"
        tim = time.time()
        print(pool.map(f, [add_ops for _ in range(processes)]))
        seq_runtimes.append(time.time() - tim)

plt.plot(runtimes, label='runtimes')
plt.plot(seq_runtimes, label='seq runtimes')
plt.plot((max_cores, max_cores), plt.ylim(), label='max cores')
plt.ylabel('time')
plt.xlabel('processes (p)')
plt.title('time to run p parallel sets of add operations with p processes')
plt.legend()
plt.savefig('data/temp.png')
plt.show()
