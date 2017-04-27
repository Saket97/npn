import numpy as np
import tensorflow as tf
import math
from operator import add
class Model:
    def __init__(self,data,target,word_dim=8000,hidden=128):
        self.input = data
        self.target = target
        self.c_square = tf.constant(math.pi)
        self.alpha = tf.constant(8-4*math.sqrt(2.0))
        self.Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
        self.hidden_dim = hidden
        self.word_dim = word_dim

    def transformFunction(x,y):
        return x,y

    def transformFunctionInverse(x,y):
        return x,y

    def npn_ops(self,weights,biases,i):
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
        state = (map(add,npn_ops(U_weights,U_biases,inputs)[0],npn_ops(W_weights,W_biases,state_old)[0])
        state = [tf.tanh(x) for x in state]
        out = npn_ops(W_weights,W_biases,state)[0]
        return out,state

    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def prediction(self,input):
        with tf.variable_scope('word_embedding'):
            w_word = tf.get_variable(name = 'w_word', shape =[vocab_size,embedding_size], initializer =tf.truncated_normal_initi
            b_word = tf.get_variable(name = 'b_word', shape =[1,embedding_size], initializer =tf.constant_initializer(0.1))

        length_sent = self.length(input)

        i = 0
        l = 0
        out_list = []

        for sent in tf.unstack(input):
            state = [tf.zeros([self.hidden_dim]),tf.zeros([self.hidden_dim])]
            j = 0
            l = 0
            sent_embed = tf.matmul(sent,w_word)+b_word
            cond_1 = lambda index_word,state,out: tf.less(index_word,length_sent[i])
            def body_1(index_word,state,out):
                global l,j
                word = sent_embed[j]
                if l==0:
                    with tf.variable_scope('rnn_1'):
                        out, state = self.rnn_cell(word,state)
                    l=1
                else:
                    with tf.variable_scope('rnn_1',reuse=True):
                        out, state = self.rnn_cell(word,state)
                j+=1
                index_word = index_word + tf.constant(1)
            return index_word,state,out
            index_word,state,out = tf.while_loop(cond_1, body_1, [index_word,state,out],swap_memory = True)
            out_list.append(tf.nn.softmax(out))
            i+=1

    def optimize(self):
        return

    def error(self):l
        return
