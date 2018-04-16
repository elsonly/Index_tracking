import tensorflow as tf
import numpy as np


class CNN:
    def __init__(self, n_features, n_assets, window,
        learning_rate):

        self.sess = tf.Session()
        self.n_features = n_features
        self.n_assets = n_assets
        self.window = window
        self.lr = learning_rate

        self._build_network()
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self):
        with tf.name_scope('inputs'):
            self.S = tf.placeholder(tf.float32,
                                    [None, self.n_assets, self.window, self.n_features],
                                    name='observations')

            self.y = tf.placeholder(tf.float32, shape=[None, self.n_assets], name='returns')
            self.I = tf.placeholder(tf.float32, shape=[None,], name='index')

            #self.previous_w = tf.placeholder(tf.float32, shape=[None, self.n_assets], name='previous_w')

        with tf.name_scope('layers'):
            c1 = tf.layers.conv2d(self.S,
                                  filters=32,
                                  kernel_size=(self.n_assets,2),
                                  strides=(1, 1),
                                  padding='valid', 
                                  data_format='channels_last', 
                                  activation=tf.nn.relu,
                                  name='c1')
            #input 4-D Tensor [batch, height, width, in_channels            
            #width = c1.get_shape()[2]

            c2 = tf.layers.conv2d(c1,
                      filters=32,
                      kernel_size=(1, 2),
                      strides=(1, 1),
                      padding='valid',
                      data_format='channels_last', 
                      activation=tf.nn.relu,
                      name='c2')


            flatten = tf.contrib.layers.flatten(c2)

            fc1 = tf.layers.dense(inputs=flatten,
                                units=32,
                                activation=tf.nn.relu,
                                name='fc1')
            #dropout = tf.layers.dropout(inputs=dense, rate=0.4)
            self.out = tf.layers.dense(inputs=fc1,
                                units=self.n_assets,
                                activation=tf.nn.softmax,
                                name='out')
            
        with tf.name_scope('loss'):
            
            self.pred_ret = tf.reduce_sum(self.y * self.out, axis=1)
            print(self.pred_ret.get_shape())
            self.loss = tf.losses.mean_squared_error(self.I, self.pred_ret)
            print(self.loss.get_shape())

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



            


                                    



