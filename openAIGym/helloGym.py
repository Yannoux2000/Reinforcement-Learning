import gym
import time

env = gym.make('CartPole-v0')
env.reset()

print env.observation_space
print env.action_space.n
# print env.observation_space.high
# print env.action_space.high

for i in range(1000):
	env.render()
	obs,reward,done,_ = env.step(env.action_space.sample())
	time.sleep(0.02)
	if(i%1000==0):
		print env.action_space.sample()

	if done:
		break
