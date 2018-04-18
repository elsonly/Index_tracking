import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from agent import Agent
from config import *
from network import CNN
from auto_train import auto_train

#tf.reset_default_graph()
NNAgent = Agent(config)

epoch = 0
while True:
    print('epoch: ', epoch)
    NNAgent, LOSS_train, LOSS_test = auto_train(NNAgent, config)

    # save model
    model_name = config['index'] + '_' + str(NNAgent.network.n_assets)
    NNAgent.save_model( model_name, epoch)

    # save assets
    path = './model/'+ model_name+ '/' + model_name +'_assets'
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(NNAgent.dm.assets, f) 

    # weight
    _w = NNAgent.weight( NNAgent.memory_val['S'] )
    weight_index = _w[0]>1e-4

    if weight_index.sum() == weight_index.shape[0]:
        print('roll_train done')
        break
    else:        
        epoch += 1
        # reset memory
        NNAgent.reset_Memory(weight_index)

        # reset n_assets
        assets = NNAgent.dm.assets.copy()
        for i in range(weight_index.shape[0]):
            if not weight_index[i]:
                assets.remove(NNAgent.dm.assets[i])
        NNAgent.dm.assets = assets
        NNAgent.network.n_assets = len(NNAgent.dm.assets)
        print('number of assets: ', NNAgent.network.n_assets)

        # reset network
        tf.reset_default_graph()
        NNAgent.network = CNN(n_features=NNAgent.memory['S'].shape[-1],
                         n_assets=NNAgent.memory['S'].shape[1],
                         config=config)
