import numpy as np
import tensorflow as tf


#from network import CNN
from network_new import CNN
from preprocessing import DataManager


class Agent:
    def __init__(self, config):
        self.batch_size = config['batch_size']

        self.dm = DataManager(index=config['index'],
                start=config['start'],
                end=config['end'],
                rng=config['rng'],
                standard=config['standard'],
                window=config['window'])

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
        S,y,I = self.dm.get_data()
        self.memory = {'S':S,'y':y,'I':I}
        self.memory_size = I.shape[0]

    def next_batch(self):
        # sample
        sample_index = np.random.randint(self.batch_size, self.memory_size)
        b_S = self.memory['S'][sample_index-self.batch_size:sample_index]
        b_S = self.dm.normalize(b_S)
        b_y = self.memory['y'][sample_index-self.batch_size:sample_index]
        b_I = self.memory['I'][sample_index-self.batch_size:sample_index]

        return b_S, b_y, b_I

    
    
    def loss(self, S, y, I):
        loss = self.network.sess.run(self.network.loss,
                    feed_dict={
                        self.network.S: S,
                        self.network.y: y,
                        self.network.I: I})
        return loss






