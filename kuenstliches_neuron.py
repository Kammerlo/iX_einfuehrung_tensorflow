import tensorflow as tf

num_inputs = 10
num_outputs = 1
x = tf.placeholder(tf.float32,shape=[None,num_inputs],name="x")
# Gewichte
w1 = tf.Variable(tf.random_uniform(shape=[num_inputs,num_outputs]),name="w1")
# Schwellwert
b1 = tf.Variable(tf.random_uniform(shape=[num_outputs]), name='b1')
# Multiplizierte Gewichte und addierter Schwellwert
sum = tf.add(tf.matmul(x,w1),b1)
# Aktivierungsfunktion, hier relu (Rectified linear units)
# Neuron mit 10 Eingängen und 1 Ausgang
neuron = tf.nn.relu(sum)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Auswertung des Neurons mit zufalls werten
    print(sess.run(neuron,feed_dict={x: [[1,2,3,4,5,6,7,8,9,10]]})) # Ergebnis des Neurons