#We have taken help from example code from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py 
from math import pi
import math
import tensorflow as tf

c_square = tf.constant(pi)
Alpha = tf.constant(8-4*math.sqrt(2.0))
Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
num_hidden_units = 800
dim_inputs = 784
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)


def transformFunction(x,y):

    return x,y

def transformFunctionInverse(x,y):

    return x,y

def declareVariables():

    #For First Layer
    o_m.append("dummy")
    o_d.append("dummy")
    o_s.append("dummy") #Dummy Variables to keep equations consistent 
    o_c.append("dummy") # Dummy Variable to eep equations consistent
    W_m.append("dummy")
    W_s.append("dummy")
    #Parameters
    a_c.append( tf.placeholder(tf.float32, shape= (dim_inputs,None)))
    a_d.append( tf.placeholder(tf.float32, shape= (dim_inputs,None)))

    W_m.append(tf.Variable( tf.random_normal([800,784], mean = 0 ,stddev =1 )))
    W_s.append(tf.Variable( tf.random_normal([800,784], mean = 0 ,stddev =1 )))
    o_s.append(tf.Variable( tf.random_normal([800,1], mean = 0 ,stddev =1 )))
    o_m.append(tf.Variable( tf.random_normal([800,1], mean = 0 ,stddev =1 )))
    o_c.append(tf.Variable( tf.random_normal([800,1], mean = 0 ,stddev =1 )))
    o_d.append(tf.Variable( tf.random_normal([800,1], mean = 0 ,stddev =1 )))
    
a_m = []
a_s = []
a_c = []
a_d = []
o_c = []
o_d = []
W_m = []
b_m = []
W_s = []


#L is the number of Hidden Layers here
L =2
batch_size = 32

for l in range(1,L+1):

    #declaring the tensorflow variables

    #Equation 1
    a_m[l-1],a_s[l-1] = transformFunction( a_c[l-1], a_d[l-1]) 
    o_m[l] =  tf.matmul(W_m[l], a_m[l-1]) + b_m[l] 

    #Equation 2
    term_1 = tf.matmul(W_s[l], a_s[l-1])
    term_2 = tf.matmul( tf.mul( W_m[l], W_m[l] ), a_s[l-1] ) 
    term_3 = tf.matmul(W_s[l-1], tf.multiply( a_m[l-1], a_m[l-1]))
    o_s[l] =  term1 + term_2+ term_3 + b_s[l]

    #Equation 3
    o_c[l],o_d[l] = transformFunctionInverse(o_m[l],o_s[l])

    #Equation 4
    o_d_term = tf.multiply(c_square, o_d[l])
    root_term = tf.sqrt(tf.add(0.25,o_d_term))

    a_m_sigm_term = tf.sigmoid( tf.divide(o_c[l], root_term))
    a_m[l] =  tf.multiply(2,a_m_sigm_term) +1

    a_s_sigm_term = Alpha*(o_c[l]+Beta)/tf.sqrt(1+c_square*tf.square(Alpha)*o_d[l])

    a_s[l] = 4*a_s_sigm_term - tf.square(a_m[l]) -2*a_m[l]-1
    a_c[l], a_d[l] = transformFunctionInverse(a_m[l], a_s[l])

# Adjusting the last layer W_m and W_s
W_m[L] = tf.Variable( tf.random_normal([10,num_hidden_units],mean = 0, stddev=1))
W_s[L] = tf.Variable( tf.random_normal([10,num_hidden_units], mean=0, stddev=1))

#Equation 1
a_m[L-1],a_s[L-1] = transformFunction( a_c[L-1], a_d[L-1]) 
o_m[L] =  tf.matmul(a_m[L-1],W_m[L-1]) + b_m[L] 

#Equation 2
term_1 = tf.matmul(W_s[L],a_s[L-1])
term_2 = tf.matmul( tf.mul( W_m[L-1], W_m[L-1] ), a_s[L-1] ) 
term_3 = tf.matmul(w_s[L],tf.multiply( a_m[L-1], a_m[L-1]))
o_s[L] =  term1 + term_2+ term_3 + b_s[L]

o_d_term = tf.multiply(c_square, o_d[l])
root_term = tf.sqrt(tf.add(1,o_d_term))

a_m[L] = tf.sigmoid( tf.divide(o_c[L], root_term))
#Equation 3
o_c[L],o_d[L] = transformFunctionInverse(o_m[L],o_s[L])


#defining the loss terms and final loss
epsilon_tensor = epsilon*tf.ones([10,1])
loss = 0.5*(tf.multiply( epsilon_tensor/o_d[L], tf.ones([10,1])) + 1/o_c[L]*(o_c[L] - y) - K + tf.log( o_d[L])*tf.ones([10,1]) - K*tf.log(epsilon))

optimizer = tf.train.AdamOptimizer()

optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_examples = int(mnist.train.num_examples)
        # Loop over all batches
        for i in range(total_examples):
            x_train, y_train = mnist.train.next_batch(total_examplgs)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run(loss, feed_dict={x: x_train, y: y_train})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(o_c[L], 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
