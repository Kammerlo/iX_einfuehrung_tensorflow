import tensorflow as tf
from utils import create_dnn_model,create_batch
import matplotlib.pyplot as plt
import numpy as np

print("Start Train")
learning_rate = 1e-3
# Hier werden alle Tensorboard logs geschrieben.
tensorboard_path = "logs/"
graph = tf.Graph()
batch_size =  100000
epochs = 500
with graph.as_default():
    x, y, net,weights,biases = create_dnn_model([2,20,20, 2])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_result = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1)) # Vergleich zwischen Prediction und label
    accuracy = tf.reduce_mean(tf.cast(correct_result, tf.float32)) # Genauigkeit über den Batch

    plot_accuracy = []
    plot_loss = []

# Merge summaries for training
with graph.as_default():
    for i in range(len(weights)):
        tf.summary.histogram("W{}".format(i),weights[i])
        tf.summary.histogram("B{}".format(i),biases[i])
    tf.summary.scalar("accuracy_train", accuracy)
    train_merge_op = tf.summary.merge_all() # merged alle summaries


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(tensorboard_path + "/train/", graph=graph)
    for i in range(epochs):
        X, Y = create_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
        # X,Y = create_batch(batch_size)
        summary,a,c = sess.run([train_merge_op,accuracy,cost],feed_dict={x: X,y:Y})
        train_writer.add_summary(summary,i)
        # plot_accuracy.append(a)
        # plot_loss.append(c)
        print("Epoch: {} of {} - Train loss: {:1.3} Train Acc: {:1.3}".format(i + 1, epochs, round(c,3),round(a,3)))




    print("-------- Plot --------")
    testDataSize = 500
    X, Y = create_batch(testDataSize)
    out = sess.run(net, feed_dict={x: X})
    pos = []
    neg = []
    for i in range(testDataSize):
        p = X[i]
        pred = out[i]
        arg = np.argmax(pred)
        if arg == 0:
            pos.append(p)
        else:
            neg.append(p)
            
    t = np.arange(-50.0,50.0,0.1)
    s = np.add(np.subtract(np.power(t,2),np.multiply(t,4)),2)
    plt.plot(t,s)
    plt.scatter([item[0] for item in pos],[item[1] for item in pos],marker="x")
    plt.scatter([item[0] for item in neg],[item[1] for item in neg],marker=".")
    plt.show()



    print("Done")