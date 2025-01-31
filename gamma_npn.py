from math import pi
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#Gamma_NPN
# constants declaration
num_hidden_units = 800
num_train = 55000
train_epoch = 10
dim_inputs = 784
units = [784,800,800,10]
L = 3
r = 1.0
tau = 0.1
batch_size = 128


mnist = input_data.read_data_sets("data/", one_hot=True)


graph = tf.Graph()
with graph.as_default():
    
    c_square = tf.constant(pi)
    Alpha = tf.constant(8-4*math.sqrt(2.0))
    Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
    def transformFunction(x,y):
        return x/y, x/y**2

    def transformFunctionInverse(x,y):
        return x**2/y,x/y

    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape))
        #initial = tf.random_uniform(shape, -1.0,1.0)
        #return tf.Variable(initial)

    image_batch = tf.placeholder(tf.float32,[batch_size,784])
    label_batch = tf.placeholder(tf.float32,[batch_size,10])

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
        a_s.append(1)
        a_c.append(0)
        a_d.append(0)
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

    def forward_pass(data):
        a_m[0] = data
        # L covers the output layer also, even though a_m[L] is not required
        # equations
        for l in range(1, L+1):
            o_m[l] = tf.matmul(W_m[l], a_m[l-1]) + b_m[l]
            o_s[l] = tf.matmul(W_s[l], a_s[l-1]) + tf.matmul(W_m[l]*W_m[l], a_s[l-1]) + tf.matmul(W_s[l], a_m[l-1]*a_m[l-1]) + b_s[l]
            o_c[l], o_d[l] = transformFunctionInverse(o_m[l], o_s[l])
            a_m[l] =tf.abs(r*(1 - tf.pow(tf.abs((o_d[l])/(o_d[l]+tau)),o_c[l])))
            a_s[l] =tf.abs(tf.pow(r,2)*( tf.pow(tf.abs((o_d[l])/(o_d[l]+2*tau)),o_c[l])- tf.pow(tf.abs((o_d[l])/(o_d[l]+tau)),2*o_c[l])))
            #print("val l : ",l,len(a_c),len(a_d),len(a_m),len(a_s))
            a_c[l], a_d[l] = transformFunctionInverse(a_m[l], a_s[l])
        return o_m[l],a_m[l]


    predictions=[]
    for image in tf.unstack(image_batch):
        image = tf.reshape(image,shape=[units[0],1])
        predict,a = forward_pass(image)
        predictions.append(tf.reshape(predict,shape= [1,units[3]]))
    predictions = tf.reshape(tf.stack(predictions),shape = [batch_size,10])
    #print("Pred shape:",predictions)
    image = forward_pass(image)
    correct_prediction = tf.equal(tf.argmax(label_batch,1), tf.argmax(predictions,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_batch,logits= predictions))
    #cross_entropy = (tf.reduce_mean(predictions-label_batch))
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
    saver = tf.train.Saver()

#init = tf.global_variables_initializer()
#new_saver = tf.train.import_meta_graph('model-gamma.ckpt.meta')
with tf.Session(graph=graph) as sess:
    print "Running Session"
    sess.run(tf.global_variables_initializer())
    print "Session initialized"
    #new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    for epoch in range(train_epoch):
        for step in range(num_train/batch_size):
            x_train, y_train = mnist.train.next_batch(batch_size)
            pred, acc,loss,_= sess.run([predictions,accuracy,cross_entropy,optimizer],feed_dict={image_batch:x_train,label_batch:y_train})
            if step%10 == 0:
                print("Epoch:",epoch," Step:",step," acc: ",acc," loss:",loss)
        print("Saving Model")
        save_path = saver.save(sess, "model-gamma.ckpt")
        print("Model saved in file: %s" % save_path)
        accuracy_total =0
        for step in range(10000/batch_size):
            x_test,y_test = mnist.test.next_batch(batch_size)
            pred, acc,loss= sess.run([predictions,accuracy,cross_entropy],feed_dict={image_batch:x_test,label_batch:y_test})
            acc_total+=acc
            print("Epoch:",epoch," Step:",step," acc_test: ",acc_total/(step+1)," loss:",loss)
