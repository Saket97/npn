import functools
import numpy as np
import tensorflow as tf
import math
from operator import add


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length



class Model:

    def __init__(self,data,target,word_dim=8000,hidden=128):
        self.input = data
        self.target = target
        self.c_square = tf.constant(math.pi)
        self.alpha = tf.constant(8-4*math.sqrt(2.0))
        self.Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
        self.hidden_dim = hidden
        self.word_dim = word_dim
        self.embedding_size = 256
        
        self.optimize
        self.prediction
        self.rnn_cell
        self.npn_ops
        #self.length
        #self.transformFunction
        #self.transformFunctionInverse
        self.error


    def transformFunction(x,y):
        return x,y

    def transformFunctionInverse(x,y):
        return x,y

    
    def npn_ops(self,weights,i):
        o = [None]*4
        a = [None]*4
        o[0] = tf.matmul(weights[0,:,:],i[0])
        o[1] = tf.matmul(weights[1,:,:], i[1]) + tf.matmul(weights[0,:,:]*weights[0,:,:], i[1]) + tf.matmul(weights[1,:,:], i[0]*i[0])
        o[2], o[3] = self.transformFunctionInverse(o[0], o[1])
        tmp = o[2]/((1+tf.abs(self.c_square*o[3]))**0.5)
        a[1] = tf.sigmoid(tmp)
        tmp = self.Alpha*(o[2]+self.Beta)/((1+tf.abs(self.c_square*self.Alpha*self.Alpha*o[3]))**0.5)
        a[1] = tf.sigmoid(tmp) - a[0]*a[0]
        a[2], a[3] = self.transformFunctionInverse(a[0], a[1])
        return [o,a]

    def rnn_cell(self,inputs,state_old):

        U_weights = tf.get_variable("U_weights",[2,self.hidden_dim,self.word_dim],tf.random_normal_initializer())
        #U_biases = tf.get_variable("U_biases",[2,self.hidden_dim],tf.random_normal_initializer())
        V_weights = tf.get_variable("V_weights",[2,self.word_dim,self.hidden_dim],tf.random_normal_initializer())
        #V_biases = tf.get_variable("V_biases",[2,self.word_dim],tf.random_normal_initializer())
        #W_biases = tf.get_variable("W_biases",[2,self.hidden_dim,self.hidden_dim],tf.random_normal_initializer())
        W_weights = tf.get_variable("W_weights",[2,self.hidden_dim,self.hidden_dim],tf.random_normal_initializer())
        state = [None]*2
        state = map(add,npn_ops(U_weights,inputs)[0],npn_ops(W_weights,state_old)[0])
        state[0] = tf.tanh(state[0])
        state[1] = tf.tanh(state[1])
        out = npn_ops(V_weights,state)[0]
        return out,state

    
    

    @lazy_property
    def prediction(self):
        #with tf.variable_scope('word_embedding'):
        #    w_word = tf.get_variable(name = 'w_word', shape =[self.word_dim,self.embedding_size], initializer = tf.truncated_normal_initializer())
        #    b_word = tf.get_variable(name = 'b_word', shape =[1,self.embedding_size], initializer =tf.constant_initializer(0.1))
        length_sent = length(self.input)
        #with tf.variable_scope('word_vocab'):
            #w_vocab = tf.get_variable(name = 'w_vocab', shape =[self.embedding_size,self.word_dim], initializer = tf.truncated_normal_initializer())
            #b_vocab = tf.get_variable(name = 'b_vocab', shape =[1,vocab_size], initializer =tf.constant_initializer(0.1))
        i = 0
        l = 0
        out_list = []


        for sent in tf.unstack(self.input):
            index_word = tf.constant(0)
            state = [tf.zeros([self.hidden_dim]),tf.zeros([self.hidden_dim])]
            out = tf.zeros([self.word_dim],dtype=tf.float32)
            j = 0
            l = 0
            #sent_embed = tf.matmul(sent,w_word)+b_word
            cond_1 = lambda index_word,state,out: tf.less(index_word,length_sent[i])
            out_list_2 = []
            def body_1(index_word,state,out):
                global l,j, out_list_2
                word = sent[j]
                if l==0:
                    with tf.variable_scope('rnn_1'):
                        out, state = self.rnn_cell(word,state)
                    l=1
                else:
                    with tf.variable_scope('rnn_1',reuse=True):
                        out, state = self.rnn_cell(word,state)
                #out_vocab = self.npn_ops(w_vocab, out)[0]
                j+=1
                index_word = index_word + tf.constant(1)
                print ("Out",out[0])
                out_list_2.append(out[0])
            return index_word,state,out[0]
            index_word,state,out = tf.while_loop(cond_1, body_1, [index_word,state,out],swap_memory = True)
            out_list.append(tf.cast(out_list_2,dtype=tf.float32))
            i+=1

        return out_list

    @lazy_property
    def optimize(self):

        predictions = self.prediction
        print ("Predictions",predictions)
        predictions = tf.stack(predictions)


        loss = tf.nn.softmax_cross_entropy_with_logits(labels = tf.cast(self.target,dtype=tf.float32),logits = predictions)
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        return loss


    
    def error(self):
        return


def train(input_file,target_file,epochs,batch_size):
    X = np.load(input_file)
    Y = np.load(target_file)
    embedding_size = 256
    num_train = X.shape[0]
    input = tf.placeholder(tf.float32, [None, 791,8000])
    target = tf.placeholder(tf.float32, [None, 791,8000])
    m = Model(input,target)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        for step in range(num_train/batch_size):
            offset = (step * batch_size) % (X.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = X[offset:(offset + batch_size), :].astype(int)
            batch_labels = Y[offset:(offset + batch_size), :].astype(int)
            batch_data = np.zeros(list(batch_data.shape) + [8000])
            batch_labels = np.zeros(list(batch_labels.shape) + [8000])
            loss = sess.run(m.optimize,feed_dict={input:batch_data,target: batch_labels})
            print("Epoch:",epoch," Step:",step," loss:",loss)
    return