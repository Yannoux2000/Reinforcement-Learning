import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'	

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def fc_layer(input,channel_in,channel_out,name_scope="fcLayer"):
	with tf.name_scope(name_scope):
		w = tf.Variable(tf.zeros([channel_in,channel_out]),name="Weights")
		b = tf.Variable(tf.zeros([channel_out]),name="Bias")
		y = tf.nn.softmax(tf.matmul(input,w) + b)

		tf.summary.histogram("Weights", w)
		tf.summary.histogram("Bias", b)
		tf.summary.histogram("Output", y)

		return y

#784 comme les 28x28 pixels des images qui sont present dans les fichiers visuel

x = tf.placeholder(tf.float32,[None,784],name="X")
y_ = tf.placeholder(tf.float32, [None, 10],name="Label")
#dans ces cas la metre null permet de rendre indefini le nombre de valeur a entrer simultanement.
#ceci ce reperecute indegniablement sur le calcule de l'erreur

y = fc_layer(x, 784, 10)

# with tf.name_scope("NeuralNet"):
# 	w = tf.Variable(tf.zeros([784,10]),name="Weights")
# 	b = tf.Variable(tf.zeros([10]),name="Bias")
# 	y = tf.nn.softmax(tf.matmul(x,w) + b)

# 	tf.summary.histogram("Weights", w)
# 	tf.summary.histogram("Bias", b)
# 	tf.summary.histogram("Output", y)

with tf.name_scope("Cross_entropy"):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
tf.summary.scalar("Cross entropy", cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/MNIST/3")
writer.add_graph(sess.graph)

for i in range(2000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	if(i%5==0):
		s = sess.run(merged_summary,feed_dict={x: batch_xs, y_: batch_ys})
		writer.add_summary(s, i)


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

xs,ys = mnist.train.next_batch(1)
ximage = xs.reshape(28,28)
cv2.imshow('Image', ximage)
returned = sess.run(y,{x:xs})

guess_format = []
i = 0
for x in returned[0]:
	guess_format.append({i : (x/0.01)})
	i+=1


for a in guess_format:
	print a

cv2.waitKey(0)
cv2.destroyAllWindows()