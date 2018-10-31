import tensorflow as tf


# Um irritierende Ausgaben von Tensorflow zu vermeiden, wird alles in einer Session ausgeführt.
# Es geht aber natürlich auch wie im Heft dargestellt mit tf.Session().run(...)

with tf.Session() as sess:
    print("----------placeholder example----------")
    # Bei einem placeholder ist dessen Typ anzugeben.
    # Die Form (shape) des Placeholders kann angegeben werden, muss jedoch nicht.
    placeholder = tf.placeholder(dtype=tf.float32,name="placeholder",shape=None)
    print(sess.run(placeholder,feed_dict={placeholder:5}))  # ==> 5.0
    print(sess.run(placeholder,feed_dict={placeholder:[5,5]}))# ==> [5.0, 5.0]

    print("----------constant example----------")
    # Konstanter Tensor mit dem Wert 0.5
    constant = tf.constant(5,shape=[1],name="constant")
    print(sess.run(constant)) # ==> [5]
    constant = tf.constant([1,2],shape=[3],name="constant")
    # Auffüllen der Konstanten mit der letzten Stelle
    print(sess.run(constant)) # ==> [1 2 2]


    print("----------variable example----------")
    # Variable ist noch nicht initialisiert.
    fixed = tf.Variable(5,name="fixed")
    random = tf.Variable(tf.random_uniform(shape=[1],minval=0,maxval=10),name="random")
    # Jetzt sind alle Variablen initialisiert
    sess.run(tf.global_variables_initializer())
    print(sess.run(fixed)) # ==> 5
    print(sess.run(random)) # ==> [6.940336]