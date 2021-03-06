import tensorflow as tf

from agent import Agent
from config import *

tf.reset_default_graph()

NNAgent = Agent(config)
LOSS_train = []
LOSS_test = []

print('start trainning ...')
for step in range(config['training_steps']):
    b_S, b_y, b_I = NNAgent.next_batch()
    NNAgent.learn(b_S,b_y,b_I)

    if (step % 1000 ==0) and (step!=0) :
        loss_train = NNAgent.loss( NNAgent.memory['S'],
                            NNAgent.memory['y'],
                            NNAgent.memory['I'])

        loss_test = NNAgent.loss( NNAgent.memory_test['S'],
                            NNAgent.memory_test['y'],
                            NNAgent.memory_test['I'])
        print('step:',step,
            'trainning loss:', loss_train,
            'testing loss:', loss_test)
        #if len(LOSS_test)>1:
        #    if LOSS_test[-1]<loss_test:
        #        break
        LOSS_train.append(loss_train)
        LOSS_test.append(loss_test)










