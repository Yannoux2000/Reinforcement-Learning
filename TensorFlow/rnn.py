import tensorflow as tf

episodes = 2000

def NN(input,chan_in,chan_out,name="NeuralNet"):
	w = tf.Variable(tf.random_normal([chan_in*3+chan_out*2,chan_out],mean=1,stddev=0.8),name="W")
	b = tf.Variable(tf.zeros([chan_out]),name="B")

	#declaration des registres
	outputm1 = tf.Variable(tf.zeros([1,chan_out]),dtype=tf.float32,name="Output_Memory")
	inputm1 = tf.Variable(tf.zeros([1,chan_in]),dtype=tf.float32,name="Input_Memory")
	outputm2 = tf.Variable(tf.zeros([1,chan_out]),dtype=tf.float32,name="Output_Memory")
	inputm2 = tf.Variable(tf.zeros([1,chan_in]),dtype=tf.float32,name="Input_Memory")

	act = tf.matmul(tf.concat([input,inputm1,inputm2,outputm1,outputm2],1), w) + b

	#on assign les variables precedentes au registres
	update_output1 = tf.assign(outputm2,outputm1)
	update_input1 = tf.assign(inputm2,inputm1)
	update_output2 = tf.assign(outputm1,act)
	update_input2 = tf.assign(inputm1,input)

	update = tf.group(*[update_output1,update_output2,update_input1,update_input2]) #regroupement des operation d'update des registres

	#on assign des zeros au registres
	reset_output2 = tf.assign(outputm2,tf.zeros([1,chan_out]))
	reset_input2 = tf.assign(outputm2,tf.zeros([1,chan_in]))
	reset_output1 = tf.assign(outputm1,tf.zeros([1,chan_out]))
	reset_input1 = tf.assign(outputm1,tf.zeros([1,chan_in]))

	reset = tf.group(*[reset_output1,reset_output2,reset_input1,reset_input2]) #regroupement des operation de reset des registres

	return act,update,reset

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

act,update,reset = NN(x, 1, 1)

loss = tf.squared_difference(act,y)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

N = 5
batch_x = [[0],[0],[0],[0],[0], [1],[1],[1],[1],[1]]
batch_y = [[0],[0],[0],[0],[0], [0.5],[0.75],[0.875],[0.9375],[0.9688]]

print "Avant Training : "
for index in range(len(batch_x)):
	print sess.run(act,{x:[batch_x[index]]})
	sess.run(update,{x:[batch_x[index]]})

sess.run(reset)

print "Training ..."

for episode in range(episodes):
	for index in range(len(batch_x)):
		sess.run(train,{x:[batch_x[index]],y:[batch_y[index]]})
		sess.run(update,{x:[batch_x[index]]})
		# if episode%200==0:
		# 	# print index
		# 	print sess.run(act,{x:[batch_x[index]]})
	sess.run(reset)

print "Done Training : "
for index in range(len(batch_x)):
	print sess.run(act,{x:[[1]]})
	sess.run(update,{x:[[1]]})
