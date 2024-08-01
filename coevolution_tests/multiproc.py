from multiprocessing import Pool
# pathos has same issue
# from pathos.multiprocessing import Pool
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy


def iniitalize_model(t):
    model = PPO(policy=MlpPolicy, env='CartPole-v1', batch_size=128, n_steps=128)
    return 1


# TODO: with the following line uncommented, it does not work
iniitalize_model(1)

with Pool(processes=1) as pool:
    test = pool.map(iniitalize_model, [None for _ in range(1)])
print('done')
