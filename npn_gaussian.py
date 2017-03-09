from math import pi

c_square = tf.constant(pi)
Alpha = tf.constant(8-4*tf.sqrt(2))
Beta = tf.constant(-0.5*tf.log(tf.sqrt(2)+1))


a_m = []
a_s = []
o_m = []
o_s = []


a_m.append() = tf.placeholder(tf.random_normal([784,1],mean= 0, stddev=1))
a_s.append() = tf.placeholder(tf.random_normal([784,1],mean= 0, stddev=1))
o_m.append() = tf.placeholder(tf.random_normal([784,1],mean= 0, stddev=1))
o_s.append() = tf.placeholder(tf.random_normal([784,1],mean= 0, stddev=1))

#L is the number of Hidden Layers here

for l in range(1,L):

    #declaring the tensorflow variables
    a_m.append() = tf.Variable(tf.random_normal([800,1],mean= 0, stddev=1))
    a_s.append() = tf.Variable(tf.random_normal([800,1],mean= 0, stddev=1))
    o_m.append() = tf.Variable(tf.random_normal([800,1],mean= 0, stddev=1))
    o_s.append() = tf.Variable(tf.random_normal([800,1],mean= 0, stddev=1))

    #Equation 1
    a_m[l-1],a_s[l-1] = transformFunction( a_c[l-1], a_m[l-1]) 
    o_m[l] = tf.add( tf.matmul(a_m[l-1],W_m[l-1]) , b_m[l] )

    #Equation 2
    term_1 = tf.matmul(a_s[l-1],W_s[l])
    term_2 = tf.matmul(a_s[l-1], tf.mul( W_m[l-1], W_m[l-1] ) ) 
    term_3 = tf.matmul(tf.multiply( a_m[l-1], a_m[l-1]),W_s[l])
    o_s[l] = tf.add( term1 , term_2, term_3 , b_s[l])

    #Equation 3
    o_c[l],o_d[l] = transformFunctionInverse(o_m[l],o_s[l])

    #Equation 4
    o_d_term = tf.multiply(c_square, o_d[l])
    root_term = tf.sqrt(tf.add(0.25,o_d_term))

    a_m_sigm_term = tf.sigmoid( tf.divide(o_c[l], root_term))
    a_m[l] = tf.subtract( tf.multiply(2,a_m_sigm_term) ,1)

    a_s_sigm_term = Alpha*(o_c[l]+Beta)/tf.sqrt(1+c_square*tf.sqaure(Alpha)*o_d[l]) - tf.square(a_m)- tf.multiply(2,a_m)-1

    a_s[l] = tf.subract(4*a_s_sigm_term - tf.square(a_m[l]) -2*a_m[l],1)



