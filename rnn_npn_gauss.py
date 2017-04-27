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
        W_weights = tf.get_variable("W_weights",[2,self.hidden_dim],tf.random_normal_initializer())
        state = [None]*2
        state = (map(add,npn_ops(U_weights,U_biases,inputs)[0],npn_ops(W_weights,W_biases,state_old)[0])
        state = [tf.nn.tanh(x) for x in state]
        out = npn_ops(W_weights,W_biases,state)[0]
        return out,state


    def predicton(self):
        return


    def optimize(self):
        return

    def error(self):
        return
