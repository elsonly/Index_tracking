import numpy as np
import tensorflow as tf
import os

from network import CNN
from preprocessing import DataManager

class Agent:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.continuous_sample = config['continuous_sample']

        self.dm = DataManager(config)

        self.Memory()
        self.network = CNN(n_features=self.memory['S'].shape[-1],
                         n_assets=self.memory['S'].shape[1],
                         config=config)
        

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
        S,y,I,S_val,y_val,I_val = self.dm.get_data()
        S_test, y_test, I_test = self.dm.get_data_testing()

        self.memory = {'S':S,'y':y,'I':I}
        self.memory_size = I.shape[0]
        self.memory_val = {'S':S_val,'y':y_val,'I':I_val}
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

    def weight(self, S):
        return self.network.sess.run(self.network.out,
                         feed_dict={self.network.S: S})

    def reset_Memory(self, selected_assets):

        self.memory['S'] = self.memory['S'][:,selected_assets,:,:]
        self.memory['y'] = self.memory['y'][:,selected_assets,:]
        self.memory_val['S'] = self.memory_val['S'][:,selected_assets,:,:]
        self.memory_val['y'] = self.memory_val['y'][:,selected_assets,:]
        self.memory_test['S'] = self.memory_test['S'][:,selected_assets,:,:]

    def save_model(self, file_name, step):
        if not os.path.exists("./model"):
            os.mkdir('model')
        if not os.path.exists("./model/"+file_name):
            os.mkdir('./model/'+file_name)

        file_name = './model/'+ file_name +'/' +file_name + '.ckpt'
        print('save_model',file_name)
        self.network.saver.save(self.network.sess, file_name, global_step=step)
        
                    



