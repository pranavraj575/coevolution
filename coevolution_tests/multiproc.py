# from pathos.multiprocessing import Pool


from multiprocessing import Pool
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy


def learn(total_timesteps):
    model = PPO(policy=MlpPolicy, env='CartPole-v1', batch_size=128, n_steps=128)
    model.learn(total_timesteps=total_timesteps)
    # del model
    # raise Exception("HEE")
    return 1


# TODO: with the following line uncommented, it does not work
learn(1)

with Pool(processes=1) as pool:
    test = pool.map(learn, [256 for _ in range(2)])
print('done')
