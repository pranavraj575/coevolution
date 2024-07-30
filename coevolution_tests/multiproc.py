from pathos.multiprocessing import Pool

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

def learn(total_timesteps):
    model = PPO(policy=MlpPolicy, env='CartPole-v1', batch_size=128, n_steps=128)
    model.learn(total_timesteps=total_timesteps)
    # del model
    # raise Exception("HEE")
    return model


# TODO: with the following line, it does not work
learn(1)


# fix test


with Pool(processes=1) as pool:
    models = pool.map(learn, [256 for _ in range(10)])
print('done')
quit()
with Pool(processes=1) as pool:
    models2 = pool.map(learn, [256 for _ in range(10)])
quit()

