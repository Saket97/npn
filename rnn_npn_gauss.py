import numpy as np
import tensorflow as tf
import math
class Model:
    def __init__(self,data,target,size=[4,1,1]):
        self.c_square = tf.constant(math.pi)
        self.alpha = tf.constant(8-4*math.sqrt(2.0))
        self.Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
        self.size = size

    def transformFunction(x,y):
        return x,y

    def transforInverse(x,y):
        return x,y

    def npn_ops(self,weights,biases,i_m,i_s,i_c,i_d):
        o_m = tf.matmul(weights[0,:,:],i_m)+biases[1,:]
        o_s = tf.matmul(weights[1,:,:], i_s) + tf.matmul(weights[0,:,:]*weights[0,:,:], a_s[l-1]) + tf.matmul(weights[1,:,:], i_m*i_m)
        o_c, o_d = transformFunctionInverse(o_m, o_s)
        tmp = o_c/((1+tf.abs(self.c_squaree*o_d))**0.5)
        a_m = tf.sigmoid(tmp)
        tmp = Alpha*(o_c+self.Beta)/((1+tf.abs(self.c_square*self.Alpha*self.Alpha*o_d))**0.5)
        a_s = tf.sigmoid(tmp) - a_m*a_m
        a_c, a_d = transformFunctionInverse(a_m, a_s)
        return a_m,a_s,a_c,a_d

    def rnn_cell(self,inputs):
        U_weights = tf.get_variable("U_weights",self.size,tf.random_normal_initializer())
        V_weights = tf.get_variable("V_weights",self.size,tf.random_normal_initializer())
        W_weights = tf.get_variable("W_weights",self.size,tf.random_normal_initializer())
        return


    def predicton(self):
        return


    def optimize(self):
        return

    def error(self):
        return
