#We have taken help from example code from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py
from math import pi
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# constants declaration
c_square = tf.constant(pi)
Alpha = tf.constant(8-4*math.sqrt(2.0))
Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
num_hidden_units = 800
dim_inputs = 784
units = [784,800,800,10]
L = 3

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

mnist = input_data.read_data_sets("data/", one_hot=True)

def transformFunction(x,y):
    return x,y

def transformFunctionInverse(x,y):
    return x,y

# output of each neuron
o_m = ["dummy"]
o_s = ["dummy"]
o_c = ["dummy"]
o_d = ["dummy"]
# after applying activations
a_m = [] # initialied with input
a_s = [] # initialised with 0
a_c = ["dummy"]
a_d = ["dummy"]
# Weights
W_m = ["dummy"]
W_s = ["dummy"]
W_c = ["dummy"]
W_d = ["dummy"]
# bias
b_m = ["dummy"]
b_s = ["dummy"]
b_c = ["dummy"]
b_d = ["dummy"]

# declare  model
#Input layer
a_m.append(tf.placeholder(tf.float32, shape=[units[0],1]))
a_s.append(tf.zeros(shape=[units[0],1]))
for l in range(1,L+1):
    a_m.append(0)
    a_s.append(0)
    o_m.append(0)
    o_s.append(0)
    o_c.append(0)
    o_d.append(0)
    b_m.append(weight_variable([units[l],1]))
    b_s.append(weight_variable([units[l],1]))
    W_m.append(weight_variable([units[l], units[l-1]]))
    W_s.append(weight_variable([units[l], units[l-1]]))
    W_c.append(weight_variable([units[l], units[l-1]]))
    W_d.append(weight_variable([units[l], units[l-1]]))

# L covers the output layer also, even though a_m[L] is not required
# equations
for l in range(1, L+1):
    o_m[l] = tf.matmul(W_m[l], a_m[l-1]) + b_m[l]
    o_s[l] = tf.matmul(W_s[l], a_s[l-1]) + tf.matmul(W_m[l]*W_m[l], a_s[l-1]) + tf.matmul(W_s[l], a_m[l-1]*a_m[l-1]) + b_s[l]
    o_c[l], o_d[l] = transformFunctionInverse(o_m[l], o_s[l])
    tmp = o_c[l]/((1+c_square*o_d[l])**0.5)
    a_m[l] = tf.sigmoid(tmp)
    tmp = Alpha*(o_c[l]+Beta)/((1+c_square*Alpha*Alpha*o_d[l])**0.5)
    a_s[l] = tf.sigmoid(tmp) - a_m[l]*a_m[l]
    a_c[l], a_d[l] = transformFunctionInverse(a_m[l], a_s[l])

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# y = 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
