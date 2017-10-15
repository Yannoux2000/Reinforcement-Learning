import tensorflow as tf
from numpy import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def FCLayer(input,channel_in,channel_out,name="FCLayer"):
	with tf.name_scope(name):
		weight = tf.Variable(tf.zeros([channel_in,channel_out]),name="weight")
		bias = tf.Variable(tf.zeros([channel_out]),name="bias")
		layer = tf.nn.sigmoid(tf.matmul(input,weight) + bias)

		tf.summary.histogram("Output", layer)
		tf.summary.histogram("Weight", weight)
		tf.summary.histogram("Bias", bias)

	return layer

nnshape = [2,5,3]

observation = tf.placeholder(tf.float32,[None,nnshape[0]])
action = tf.placeholder(tf.float32,[None,nnshape[2]])

layer_1 = FCLayer(observation, nnshape[0], nnshape[1],name="FC_Layer_1")
layer_2 = FCLayer(layer_1, nnshape[1], nnshape[2],name="FC_Layer_2")
with tf.name_scope("Loss"):
	loss = tf.square(layer_2 - action)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

print "Starting session :"

N = 1

obs_batch = [[0,0]]*N
act_batch = [[0,0,0]]*N

x = 0

for obs in obs_batch:
	x+=1
	obs = [x,0]

y = 0
for act in act_batch:
	y-=1
	act = [y*2] * 3


sess = tf.Session()

#les Variables sont initialise
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter("/tmp/leepdearning/1")
msummaries = tf.summary.merge_all()
writer.add_graph(sess.graph)

for i in range(100000):
	sess.run(train,feed_dict={observation :obs_batch, action :act_batch})

	if(i%500)==0 :
		print "[{}]:\tguess {}".format(i,sess.run(layer_2, {observation: obs_batch ,action :act_batch}))

	if(i%10 ==0):
		s = sess.run(msummaries,{observation: obs_batch ,action :act_batch})
		writer.add_summary(s,i)
