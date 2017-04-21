class Model:
    def __init__(self,data,target):
        self.c_square = tf.constant(pi)
        self.alpha = tf.constant(8-4*math.sqrt(2.0))
        self.Beta = tf.constant(-0.5*math.log(math.sqrt(2.0)+1))
        self.variables_dict = {
                "U_weights": tf.Variable(tf.random_normal([size]),name="U_weights"),
                "V_weights": tf.Variable(tf.random_normal([size])), name="V_weights"),
                "W_weights": tf.Variable(tf.random_normal([size]),name = "W_weights")
        }

