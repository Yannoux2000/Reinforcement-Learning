import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

#INFO : 
#cet algo ressemble plus au Actor Critic

#Parser
# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument("lr",help="Learning Rate")
# parser.add_argument("il",help="Intermediate_layer")
# parser.add_argument("gamma",help="Gamma also called discount rate")
# parser.add_argument("beta",help="Representing additionnal exploration")
# args = parser.parse_args()

# if args.lr:
# 	learning_rate = float(args.lr)

# if args.il:
# 	intermediate_layer = int(args.il)

# if args.gamma:
# 	gamma = float(args.gamma)

# if args.beta:
# 	beta = float(args.beta)


#define :
Separated_Networks = 0

###ENV Var : 
env_name = 'CartPole-v0'
obs_space_size = 4
act_space_n = 2

##TESTPARAMETER
number_Episodes = 20000
max_steps = 1000

#meilleur score avec lr=0.01 il=0 gamma=0.95
##HYPERPARAMETER


intermediate_layer = 0
learning_rate = 0.1
beta = 0
gamma = 0.98

# Mauvaise Implementation d'un reseau commun

# def commun_gradient(input,in_channel,out_channel):
# 	with tf.variable_scope("Commun_NN"):
# 		with tf.name_scope("FCLayer"):
# 			w1 = tf.Variable(tf.random_normal([in_channel,20]),name="w1")
# 			b1 = tf.Variable(tf.zeros([20]),name="b1")
# 			h1 = tf.nn.relu(tf.matmul(input,w1) + b1)

# 			smry_w1 = tf.summary.histogram("w1", w1)
# 			smry_b1 = tf.summary.histogram("b1", b1)

# 		with tf.name_scope("FCLayer"):
# 			w2 = tf.Variable(tf.random_normal([20,out_channel]),name="w2")
# 			b2 = tf.Variable(tf.zeros([out_channel]),name="b2")
# 			calculate = tf.matmul(h1,w2) + b2

# 			smry_w2 = tf.summary.histogram("w2", w2)
# 			smry_b2 = tf.summary.histogram("b2", b2)

# 		smry_commun = tf.summary.merge([smry_b1,smry_w1,smry_b2,smry_w2])

# 		return calculate,smry_commun


def policy_gradient(input,obs_channel,act_channel,actions,advantages):
	#input,obs_channel,act_channel,actions,advantages
	with tf.variable_scope("policy"):

		with tf.name_scope("FCStohcasticLayer"):
			params = tf.Variable(tf.random_normal([obs_channel,act_channel],mean=1,stddev=1.0),name="policy_parameters")
			linear = tf.matmul(input,params)
			probabilities = tf.nn.softmax(linear)


			smry_params = tf.summary.histogram("params", params)

		with tf.name_scope("diffs_and_loss"):
			esperance = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
			#Policy_loss = -log(E[policy|s]) * A(s) // pas de calcule de beta pour le moment - beta*H(policy(s))
			#si beta = 0 alors aucune exploration n'est parametre
			eligibility = tf.log(esperance) * advantages + beta * tf.reduce_sum(tf.multiply(probabilities,tf.log(probabilities)))
			loss = -tf.reduce_sum(eligibility)

		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


	smry_loss = tf.summary.scalar("Policy_Losses", loss)
	smry_policy = tf.summary.merge([smry_loss,smry_params])

	return (probabilities, optimizer),smry_policy


def value_gradient(input,obs_channel,newvals):
	#input,obs_channel,act_channel,newvals
	with tf.variable_scope("value"):

		with tf.name_scope("FCLayer"):
			w1 = tf.Variable(tf.random_normal([obs_channel,10]),name="w1")
			b1 = tf.Variable(tf.zeros([10]),name="b1")
			h1 = tf.matmul(input,w1) + b1

			smry_w1 = tf.summary.histogram("w1", w1)
			smry_b1 = tf.summary.histogram("b1", b1)

		with tf.name_scope("FCLayer"):
			w2 = tf.Variable(tf.random_normal([10,1]),name="w2")
			b2 = tf.Variable(tf.zeros([1]),name="b2")
			calculate = (tf.matmul(h1,w2) + b2)

			smry_w2 = tf.summary.histogram("w2", w2)
			smry_b2 = tf.summary.histogram("b2", b2)

		with tf.name_scope("diffs_and_loss"):
			diffs = calculate - newvals
			loss = tf.nn.l2_loss(diffs)

		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


	smry_loss = tf.summary.scalar("Value_Losses", loss)
	smry_value = tf.summary.merge([smry_loss,smry_w1,smry_b1,smry_w2,smry_b2])


	
	return (calculate, optimizer),smry_value


def probs_to_action(probs):
	seed = random.uniform(0,1)
	action = 0
	probsum = probs[action]

	while seed >= probsum and probsum<1:
		action += 1
		probsum += probs[action]

	# print "probsum : {}".format(probsum)
	return action


def run_episode(env, policy_grad, value_grad, sess):
	pl_calculate, pl_optimizer = policy_grad
	vl_calculate, vl_optimizer = value_grad
	obs = env.reset()
	totalreward = 0
	epi_states = []
	epi_actions = []
	epi_advantages = []
	epi_transitions = []
	update_vals = []


	for t in xrange(max_steps):
		# calculate policy

		#preparation des obs
		obs_vector = np.expand_dims(obs, axis=0)# de [obs] vers [None,obs] pour les stacker durant l entrainement
		probs = sess.run(pl_calculate,feed_dict={observation: obs_vector})
		
		#traduction probabilites en epi_actions
		action = probs_to_action(probs[0])

		# record the agent_state
		epi_states.append(obs)
		actionblank = np.zeros(act_space_n)
		actionblank[action] = 1
		epi_actions.append(actionblank)
		
		# take the action in the environment
		old_obs = obs
		obs, reward, done, info = env.step(action)
		epi_transitions.append((old_obs, action, reward))
		totalreward += reward

		env.render()

		# #TD Learning
		# cette algo peu etre instable et provoque des oscilations a revoir ?

		newobs_vector = np.expand_dims(obs, axis=0)# de [obs] vers [None,obs] pour les stacker durant l entrainement
		nextval = sess.run(vl_calculate,feed_dict={observation: newobs_vector})[0][0]

		realval = reward + gamma * nextval
		sess.run(vl_optimizer,feed_dict={observation : obs_vector, newvals: [[realval]]})
		currentval = sess.run(vl_calculate,feed_dict={observation: obs_vector})[0][0]

		realadvantage = realval - currentval
		sess.run(pl_optimizer, feed_dict={observation: obs_vector, advantage: [[realadvantage]], pl_actions: [actionblank]})
		# #end TD

		if done:
			break

	#END OF THE EPISODE

	for index, trans in enumerate(epi_transitions):
		obs, _, reward = trans

		# calculate discounted monte-carlo return
		#monte-carlo is offline
		future_epi_transitions = len(epi_transitions) - index
		discount = 1
		realval = 0

		# calcule des valeur des epi_states
		for index2 in xrange(future_epi_transitions):
			realval += epi_transitions[index2 + index][2] * discount
			discount = discount * gamma
		
		obs_vector = np.expand_dims(obs, axis=0)
		
		#valeur utilise precedament
		currentval = sess.run(vl_calculate,feed_dict={observation: obs_vector})[0][0]

		# advantage: how much better was this action than normal
		epi_advantages.append(realval - currentval)

		# update the value function towards new return
		update_vals.append(realval)

	# update value function
	update_vals_vector = np.expand_dims(update_vals, axis=1)
	sess.run(vl_optimizer, feed_dict={observation: epi_states, newvals: update_vals_vector})

	# update policy function
	epi_advantages_vector = np.expand_dims(epi_advantages, axis=1)
	sess.run(pl_optimizer, feed_dict={observation: epi_states, advantage: epi_advantages_vector, pl_actions: epi_actions})

	if episode%10==0:
		s = sess.run(steps_summary,{observation: epi_states, newvals: update_vals_vector, advantage: epi_advantages_vector, pl_actions: epi_actions})
		writer.add_summary(s,episode)

	return totalreward

####MAIN####
# def main():

env = gym.make(env_name)

observation = tf.placeholder("float",[None,obs_space_size],name = "states") #label d'entree observations

#Policy Function
pl_actions = tf.placeholder("float",[None,act_space_n],name="actions") #label d'entrainement
advantage = tf.placeholder("float",[None,1],name="advantages") #label d'entrainement

#Value Function
newvals = tf.placeholder("float",[None,1],name = "newvals") #label d'entrainement

#Statistic Function (tensorboard)
reward = tf.placeholder("float",[100,1],name = "Rewards")

with tf.name_scope("Statistics"):
	mean_reward = tf.reduce_sum(tf.reduce_mean(reward))
	max_reward = tf.reduce_sum(tf.reduce_max(reward))


	stddev_reward = tf.sqrt(tf.reduce_mean(tf.square(reward - mean_reward)))

	smry_mean_reward = tf.summary.scalar("Mean_Reward", mean_reward)
	smry_stddev_reward = tf.summary.scalar("StdDev_Reward", stddev_reward)
	smry_max_reward = tf.summary.scalar("Max_Reward", max_reward)

# if intermediate_layer==Separated_Networks:

value_grad,smry_value = value_gradient(observation,obs_space_size,newvals)
policy_grad,smry_policy = policy_gradient(observation,obs_space_size,act_space_n,pl_actions,advantage)
steps_summary = tf.summary.merge([smry_value,smry_policy])

# else :
# 	#Commun Neural Net
# 	commun_grad,smry_commun = commun_gradient(observation,obs_space_size,intermediate_layer)

# 	value_grad,smry_value = value_gradient(commun_grad,intermediate_layer,newvals)
# 	policy_grad,smry_policy = policy_gradient(commun_grad,intermediate_layer,act_space_n,pl_actions,advantage)
# 	steps_summary = tf.summary.merge([smry_value,smry_commun,smry_policy])

rewards = [[0]] * 100

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("./TB/{}HPSearch/InterLayer{}-LearnRate{}-G{}-B{}/".format(env_name,intermediate_layer,learning_rate,gamma,beta))
episode_summary = tf.summary.merge([smry_mean_reward,smry_stddev_reward,smry_max_reward])
writer.add_graph(sess.graph)

bestreward = -5000 #concidere comme un seuil puis comme un record

for episode in xrange(number_Episodes):
	epi_reward = run_episode(env, policy_grad, value_grad, sess)
	rewards.append([epi_reward])
	rewards.pop(0)

	if episode%10==0:
		s = sess.run(episode_summary,{reward : rewards})
		writer.add_summary(s,episode)

	if epi_reward > bestreward or episode%250==0:
		print "[{}] : reward : {}".format(episode,epi_reward)

	if epi_reward> bestreward:
		bestreward = epi_reward
# t = 0