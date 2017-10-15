import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

#simple declaration de constante ici le dtype n'est pas obligatoire
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

#par concequence on obtiendra non pas un resultat mais un calcule
node3 = tf.add(node1,node2)
print("node3:", node3)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

add_then_triple = adder_node * 3

print("\n[?] : Starting session now ! \n")

#le demarage de la session represente le debut des calcules
sess = tf.Session()

print("sess.run(node3):\t", sess.run(node3))


print("sess.run(adder_node):\t",sess.run(adder_node,{a: 3,b: 4.5}))
print("sess.run(adder_node):\t",sess.run(adder_node,{a: [1, 3], b: [2, 4]}))

print("sess.run(adder_node):\t",sess.run(add_then_triple,{a: 3,b: 4.5}))
