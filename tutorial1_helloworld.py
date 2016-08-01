# -*- coding: utf-8 -*-
"""
Some random tutorials/examples in Tensorflow
"""

import tensorflow as tf
import numpy as np

# Think of tensorflow as a distributed numpy where computation of
# matrices (tensors) can be done on various devices (GPUs, CPUs)

tf.InteractiveSession()

# Some simple numpy array equivalent operations:
a = tf.ones((2,2))

b = tf.zeros((2,2))

tf.reduce_sum(a, reduction_indices=[1]).eval()

a.get_shape()

tf.reshape(a, (1,4)).eval()

c = a * b # This does not actually perform any computation

with tf.Session() as sess:
    print(sess.run(c))
    print(c.eval())
    
# Note that a and b above are constant tensor
    
W1 = tf.ones((2,2))

# Defining variable tensors in TF
W2 = tf.Variable(tf.zeros((2,2)), name="weights")

with tf.Session() as sess:
    print(sess.run(W1))
    # Variables need to be initialized
    sess.run(tf.initialize_all_variables())
    sess.run(W2)
    
# Let's write a incrementing counter in TensorFlow


state = tf.Variable(0, name="counter") # roughly saying state = 0

new_val = tf.add(state, tf.constant(1)) # roughly saying: new_val = state + 1

update = tf.assign(state, new_val) # roughly saying: state = new_val

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # << here actually sets state=0
    print(sess.run(state)) 
    for _ in range(3):
        sess.run(update)   # << here it actually performs the addition and assignment
        print(sess.run(state))

# Fetching multiple tensors in a session:
# and building a computational graph:

input1 = tf.constant(2.0)
input2 = tf.constant(3.0)
input3 = tf.constant(5.0)


intermed = tf.add(input1, input2)
mul_op = tf.mul(intermed, input3)

with tf.Session() as sess:
    result = sess.run([intermed, mul_op])
    print(result)


# Putting data from numpy into TF:

a = np.zeros((3,3))

a_t = tf.convert_to_tensor(a)

with tf.Session() as sess:
    result = sess.run(a_t)
    print(result)

# You can also use placeholders to load data to the computational graph


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.0], input2:[5.0]}))


# Variables scope:

with tf.variable_scope('foo'):
    with tf.variable_scope('bar'):
        v = tf.get_variable('v', [1])

assert v.name == 'foo/bar/v:0'

# variable scope also allows reusing variables:
# note that get_variable creates new variable each time

with tf.variable_scope('foo'):
    u = tf.get_variable('u',[1])
    
with tf.variable_scope('foo', reuse=True):
    u1 = tf.get_variable('u',[1])
    
assert u == u1

