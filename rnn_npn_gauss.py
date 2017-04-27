import functools
import numpy as np
import tensorflow as tf
import math
from operator import add


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator



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
        state = [tf.tanh(x) for x in state]
        out = npn_ops(V_weights,state)[0]
        return out,state

    
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    
    def prediction(self):
        with tf.variable_scope('word_embedding'):
            w_word = tf.get_variable(name = 'w_word', shape =[self.word_dim,self.embedding_size], initializer = tf.truncated_normal_initializer())
            b_word = tf.get_variable(name = 'b_word', shape =[1,self.embedding_size], initializer =tf.constant_initializer(0.1))
        length_sent = self.length(input)
        with tf.variable_scope('word_vocab'):
            w_vocab = tf.get_variable(name = 'w_vocab', shape =[self.embedding_size,self.word_dim], initializer = tf.truncated_normal())
            #b_vocab = tf.get_variable(name = 'b_vocab', shape =[1,vocab_size], initializer =tf.constant_initializer(0.1))
        i = 0
        l = 0
        out_list = []

        for sent in tf.unstack(self.input):
            state = [tf.zeros([self.hidden_dim]),tf.zeros([self.hidden_dim])]
            out_vocab = tf.zeros([self.word_dim])
            j = 0
            l = 0
            sent_embed = tf.matmul(sent,w_word)+b_word
            cond_1 = lambda index_word,state,out_vocab: tf.less(index_word,length_sent[i])
            def body_1(index_word,state,out_vocab):
                global l,j
                word = sent_embed[j]
                if l==0:
                    with tf.variable_scope('rnn_1'):
                        out, state = self.rnn_cell(word,state)
                    l=1
                else:
                    with tf.variable_scope('rnn_1',reuse=True):
                        out, state = self.rnn_cell(word,state)
                out_vocab = self.npn_ops(w_vocab, out)[0]
                j+=1
                index_word = index_word + tf.constant(1)
            return index_word,state,out_vocab
            index_word,state,out_vocab = tf.while_loop(cond_1, body_1, [index_word,state,out_vocab],swap_memory = True)
            out_list.append(out_vocab)
            i+=1
        return out_list

    
    def optimize(self):

        predictions = self.prediction
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.target,logits = predictions)
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        return loss


    
    def error(self):
        return


def train(input_file,target_file,epochs,batch_size):
    X = np.load(input_file)
    Y = np.load(target_file)
    
    num_train = X.shape[0]
    input = tf.placeholder(tf.float32, [None, 791])
    target = tf.placeholder(tf.float32, [None, 791])
    m = Model(input,target)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        for step in range(num_train/batch_size):
            offset = (step * batch_size) % (X.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = X[offset:(offset + batch_size), :]
            batch_labels = Y[offset:(offset + batch_size), :]
            loss= sess.run(m.optimize,feed_dict={input:batch_data,target: batch_labels})
            print("Epoch:",epoch," Step:",step," loss:",loss)
    return