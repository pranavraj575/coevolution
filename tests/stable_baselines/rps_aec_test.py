from pettingzoo.classic import rps_v2

env = rps_v2.env()
par_env = rps_v2.parallel_env()
env.reset()
par_env.reset()

print(env.agents)
print(env.agent_selection, env.last())
env.step(0)
print(env.agent_selection, env.last())
env.step(1)
print(env.agent_selection, env.last())
env.step(1)
print(env.agent_selection, env.last())

quit()
for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()  # this is where you would insert your policy

    env.step(action)

    print(termination or truncation)
