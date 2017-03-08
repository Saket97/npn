



for i in range(0,T):

    for l in range(0,L):

        a_m[l-1],a_s[l-1] = transformFunction( a_c[l-1], a_m[l-1]) 
        o_m[l] = tf.add( tf.matmul(a_m[l-1],W_m[l-1]) , b_m[l] )
        o_s[l] = tf.add( tf.matmul(a_s[l-1],W_s[l]) , tf.matmul(a_s[l-1], tf.mul( W_m[l-1], W_m[l-1] ) ), tf.matmul(tf.multiply( a_m[l-1], a_m[l-1]),W_s[l]) , b_s[l])
        o_c[l],o_d[l] = transformFunctionInverse(o_m[l],o_s[l])




