import tensorflow as tf

def get_network_from_name(name, X):
    nets = { "slide_defect_resnet" : slide_defect_resnet,
             "slide_defect_resnet_b" : slide_defect_resnet,
             "slide_defect_resnet_stride": slide_defect_resnet_stride }

    return nets[name](X)

def slide_defect_resnet(X):
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_3')
    net2 = tf.add(net1, net2c)

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_3')
    net3 = tf.add(net2, net3c)

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_3')
    net4 = tf.add(net3, net4c)

    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(net4, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_3')
    net5 = tf.add(net4, net5c)

    net6 = tf.contrib.layers.conv2d(net5, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

def slide_defect_resnet_b(X):
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1_3')
    net2 = tf.add(net1, net2c)

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2_3')
    net3 = tf.add(net2, net3c)

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3_3')
    net4 = tf.add(net3, net4c)

    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(net4, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4_3')
    net5 = tf.add(net4, net5c)

    net6 = tf.contrib.layers.conv2d(net5, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]

def slide_defect_resnet_stride(X):
    # Widen the network
    net1 = tf.contrib.layers.conv2d(X, 256, 1, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='widen')

    # Residual unit 1
    net2a = tf.contrib.layers.conv2d(net1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/1')
    net2b = tf.contrib.layers.conv2d(net2a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/2')
    net2c = tf.contrib.layers.conv2d(net2b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res1/3')
    net2 = tf.add(net1, net2c, name='res1/add')
    net2mp = tf.contrib.layers.max_pool2d(net2, 3, 2, 'SAME', scope='res1/mp')

    # Residual unit 2
    net3a = tf.contrib.layers.conv2d(net2mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/1')
    net3b = tf.contrib.layers.conv2d(net3a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/2')
    net3c = tf.contrib.layers.conv2d(net3b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res2/3')
    net3 = tf.add(net2mp, net3c, name='res2/add')
    net3mp = tf.contrib.layers.max_pool2d(net3, 3, 2, 'SAME', scope='res2/mp')

    # Residual unit 3
    net4a = tf.contrib.layers.conv2d(net3mp, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/1')
    net4b = tf.contrib.layers.conv2d(net4a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/2')
    net4c = tf.contrib.layers.conv2d(net4b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res3/3')
    net4 = tf.add(net3mp, net4c, name='res3/add')

    up1 = tf.contrib.layers.conv2d_transpose(net4, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up1')
    
    # Residual unit 4
    net5a = tf.contrib.layers.conv2d(up1, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/1')
    net5b = tf.contrib.layers.conv2d(net5a, 128, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/2')
    net5c = tf.contrib.layers.conv2d(net5b, 256, 3, 1, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='res4/3')
    net5 = tf.add(up1, net5c, name='res4/add')

    up2 = tf.contrib.layers.conv2d_transpose(net5, 256, 3, 2, 'SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='up2')

    net6 = tf.contrib.layers.conv2d(up2, 64, 3, 1, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow1')
    net = tf.contrib.layers.conv2d(net6, 1, 1, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), weights_regularizer=tf.nn.l2_loss, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='narrow')
    
    return [net]