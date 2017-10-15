import gym
import numpy as np

#HYPERPARAM
max_episode = 10000
max_steps = 200

def run_episode(env, parameters):

	observation = env.reset()
	totalreward = 0

	for t in range(max_steps):

		env.render()

		action = 0 if np.matmul(parameters,observation) < 0 else 1

		observation, reward, done, info = env.step(action)
		totalreward += reward

		if done:
			break

	return totalreward


env = gym.make('CartPole-v0')

#RECHERCHE ALEATOIRE

#	cette facon de faire fonctionne assez bien sur l'env CartPole.
#	la raison ? les reward sont positives. et permetent de differencier les parametre

#	le principe de la recherche aleatoire est proche a celui d un algorithme genetique
#	la difference c'est qu'on ne s'inspire jamais du resultat precedent pour trouver un autre.
#	donc l'apprentisage est hazardeux. mais il fini par fonctionner

bestparams = None
bestreward = 0

for episode in range(max_episode):

	parameters = np.random.rand(4) * 2 - 1
	reward = run_episode(env,parameters)

	print "[{}]:\treward = {}".format(episode,reward)

	if reward > bestreward:
		bestreward = reward
		bestparams = parameters

		# considered solved if the agent lasts 200 timesteps
		if reward == 200:
			print bestparams
			break
