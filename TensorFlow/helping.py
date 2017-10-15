import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


train_x = [[2012,12,11,20],[2011,12,21,11],[2012,8,13,4],[2012,5,4,8],[2013,2,19,4],[2011,10,10,19]]
train_y = [423,189,117,144,342,204]
train_y = np.expand_dims(train_y, axis=1)
test_x = [[2012,4,23,19],[2011,12,27,2],[2013,3,1,18],[2011,12,2,13],[2012,8,13,14],[2012,5,18,6]]
test_y = [375,99,516,159,186,123]
test_y = np.expand_dims(test_y, axis=1)

n_dim = 4

learning_rate = 0.01
training_epochs = 50
cost_history = []

X = tf.placeholder(tf.float32,[None,n_dim],name = "X")
Y = tf.placeholder(tf.float32,[None,1],name = "Y")

with tf.name_scope("Layer"):
    W = tf.Variable(tf.ones([n_dim,1]),name ="W")
    B = tf.Variable(tf.zeros([1]),name ="B")
    y_ = tf.matmul(X, W) + B
    tf.summary.histogram("Weight", W)
    tf.summary.histogram("Bias", B)

with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.square(y_ - Y))
    tf.summary.scalar("Cost", cost)

training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
writer = tf.summary.FileWriter("/tmp/helping/4")
msummaries = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        sess.run(training_step,feed_dict={X:train_x,Y:train_y})
        cost_history.append(sess.run(cost,feed_dict={X: train_x,Y: train_y}))
        # if epoch%5==0:
        s = sess.run(msummaries,{X: train_x ,Y :train_y})
        writer.add_summary(s,epoch)


    #calculate mean square error 
    print("MSE: %.4f" % sess.run(cost, feed_dict={X: test_x,Y: test_y}))
    
    #plot cost
    plt.plot(range(len(cost_history)),cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(test_y, pred_y)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
