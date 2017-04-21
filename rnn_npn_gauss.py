import numpy as np
import tensorflow as tf
import math
class Model:
    def __init__(self,data,target,size=[1]):
        self.c_square = tf.constant(math.pi)
        self.alpha = tf.constant(8-4*math.sqrt(2.0))
        self.Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
        self.size = size
    def npn_ops(self,weights,inputs):
        return

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
