import numpy as np
import tensorflow as tf
from config import *

from network import CNN
#from network_new import CNN
from preprocessing import DataManager


class Agent:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.continuous_sample = config['continuous_sample']
        self.dm = DataManager(config)

        self.Memory()

        self.network = CNN(n_features=self.memory['S'].shape[-1],
                         n_assets=self.memory['S'].shape[1],
                         window=config['window'],
                         learning_rate=config['learning_rate'])
        

    def learn(self,S,y,I):
        self.network.sess.run(self.network.train_op,
            feed_dict={
                self.network.S: S,
                self.network.y: y,
                self.network.I: I})

    def choose_action(self, S):
        w = self.network.sess.run(self.network.out, 
                                feed_dict={self.network.S: S})
        return w
        

    def Memory(self):
        S,y,I,S_test,y_test,I_test = self.dm.get_data()
        self.memory = {'S':S,'y':y,'I':I}
        self.memory_size = I.shape[0]
        self.memory_test = {'S':S_test,'y':y_test,'I':I_test}

    def next_batch(self):
        if self.continuous_sample:
            # sample
            sample_index = np.random.randint(self.batch_size, self.memory_size)
            b_S = self.memory['S'][sample_index-self.batch_size:sample_index]
            #b_S = self.dm.normalize(b_S)
            b_y = self.memory['y'][sample_index-self.batch_size:sample_index]
            b_I = self.memory['I'][sample_index-self.batch_size:sample_index]
        else:
            sample_index = np.random.randint(0, self.memory_size, self.batch_size)
            b_S = self.memory['S'][sample_index]
            #b_S = self.dm.normalize(b_S)
            b_y = self.memory['y'][sample_index]
            b_I = self.memory['I'][sample_index]

        return b_S, b_y, b_I

    
    
    def loss(self, S, y, I):
        loss = self.network.sess.run(self.network.loss,
                    feed_dict={
                        self.network.S: S,
                        self.network.y: y,
                        self.network.I: I})
        return loss






