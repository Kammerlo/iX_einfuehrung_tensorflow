import tensorflow as tf
import random
import math

# y = x^2 - 4x + 2
def create_batch(batchSize):
    input_data = []
    output_data = []
    for i in range(batchSize):
        x = random.uniform(-50, 50)
        y = random.uniform(-100,2500)
        f = math.pow(x,2) - 4 * x + 2
        out = [1.0,0.0] if f < y else [0.0,1.0]
        input_data.append([x,y])
        output_data.append(out)
    return input_data,output_data

# z.B. [10,20,20,1] ==> 10 Eingänge, 2 Hidden Layer mit 20 Neuronen und 1 Ausgang
def create_dnn_model(layerSizes,input=None, stddev=0.05, bias_weight_init=0.05, name_suffix=""):
    if input == None:
        x = tf.placeholder(dtype=tf.float32,shape=[None,layerSizes[0]],name="x")
    else:
        x = input
    y = tf.placeholder(dtype=tf.float32,shape=[None,layerSizes[len(layerSizes) - 1]],name="y")
    dnn = x
    weights = []
    biases = []

    for i in range(len(layerSizes) - 1):
        with tf.name_scope('weights'): # weights/WX
            w = tf.Variable(tf.truncated_normal([layerSizes[i],layerSizes[i + 1]],stddev=stddev),name="W" + str(i) + name_suffix)
            weights.append(w)

        with tf.name_scope('biases'): # biases/BX
            b = tf.Variable(tf.constant(bias_weight_init,dtype=tf.float32,shape=[layerSizes[i + 1]]),name="B" + str(i) + name_suffix)
            biases.append(b)
        layer_in = tf.add(tf.matmul(dnn, w), b)
        layer = tf.nn.relu(layer_in)

        dnn = layer

    return x,y,dnn,weights,biases
