import gym
import numpy as np

#HYPERPARAM
max_episode = 10000
max_steps = 500

def run_episode(env, parameters):

	observation = env.reset()
	totalreward = 0

	for t in range(max_steps):

		env.render()

		action = 0 if np.matmul(parameters,observation) < 0 else 1

		observation, reward, done, info = env.step(action)
		totalreward += reward

		if done:
			if totalreward < 200:
				break

	return totalreward


env = gym.make('CartPole-v0')

#Hill Climb

#	le hill climb partage beaucoup de similarite avec le randomsearch,
#	la principale difference est qu on modifie le meilleur pour avancer vers une solution
#	cela implique un defaut qui es, si les resultat sont indiferensable les un des autres
#	une recherche alleatoire est alors une meilleur solution

noise_scaling = 0.1  
parameters = np.random.rand(4) * 2 - 1  
bestreward = 0

for episode in xrange(max_episode):
	newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
	reward = 0
	reward = run_episode(env,newparams)

	if(episode%100==0 or reward > bestreward):
		print "[{}]:\treward = {}".format(episode,reward)
	
	if reward > bestreward:
		bestreward = reward
		parameters = newparams

		if reward == 200:
			print parameters
			break