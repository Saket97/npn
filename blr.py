#Example taken from https://github.com/blei-lab/edward/blob/master/examples/bayesian_linear_regression.py
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal

def build_toy_dataset(N,w,noise_std=0.1):
    D = len(w)
    x = np.random.randn(N,D).astype(np.float32)
    y = np.dot(x,w) + np.random.normal(0,noise_std,size=N)

    return x,y

ed.set_seed(123)

N = 40
D = 10

w_true = np.random.randn(D)
X_train,y_train = build_toy_dataset(N,w_true)
X_test,y_test = build_toy_dataset(N,w_true)

X = tf.placeholder(tf.float32, [N,D])
w = Normal(mu=tf.zeros(D),sigma= tf.ones(D))
b = Normal(mu=tf.zeros(1),sigma= tf.ones(1))
y = Normal(mu=ed.dot(X,w)+b,sigma= tf.ones(N))

#Inference

qw = Normal(mu = tf.Variable(tf.random_normal([D])),
        sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(mu = tf.Variable(tf.random_normal(([1]))),
        sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

inference = ed.KLqp({w: qw, b: qb},data = {X:X_train, y: y_train})
inference.run(n_samples=3, n_iter=1000)


#Criticism

y_post = ed.copy(y , {w: qw, b:qb })
# This is equivalent to
# y_post = Normal( mu = ed.dot(X, qw) + qb, sigma=tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error',data = {X : X_test, y_post: y_test}))
