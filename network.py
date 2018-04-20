import tensorflow as tf
import numpy as np


class CNN:
    def __init__(self, n_features, n_assets, config, restore=None):
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.n_features = n_features
        self.n_assets = n_assets

        self.window = config['window']
        self.holding_period = config['holding_period']
        self.lr = config['learning_rate']
        
        if restore:
            tf.reset_default_graph()
            self._build_network()
            model_path = tf.train.latest_checkpoint(restore)
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)
        else:
            self._build_network()
            self.sess.run(tf.global_variables_initializer())

        if config['save_model']:
            self.saver = tf.train.Saver()

    def _build_network(self):
        with tf.name_scope('inputs'):
            self.S = tf.placeholder(tf.float32,
                                    [None, self.n_assets, self.window, self.n_features],
                                    name='observations')

            self.y = tf.placeholder(tf.float32, shape=[None, self.n_assets, self.holding_period], name='returns') #(m, n_assets, holding_period)
            self.I = tf.placeholder(tf.float32, shape=[None, self.holding_period ], name='index') # (m, holding_period)

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
            
            out = tf.reshape(self.out,[-1, self.n_assets,1]) # out.shape=(m, n_assets)
                                                             # y.shape=(m, n_assets, holding_period)
                                                             # I.shape=(m, holding_period)
            
            self.pred_ret = tf.reduce_sum(self.y * out, axis=1)
            self.loss = tf.losses.mean_squared_error(self.I, self.pred_ret)
            
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


