import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from agent import Agent
from network import CNN


def auto_train(Agent, config):
    training_steps = config['training_steps']
    print_process = config['print_process']
    early_stop_number = config['early_stop']
    save_model = config['save_model']
    save_step = config['save_step']

    LOSS_train = []
    LOSS_test = []

    print('start trainning ...')
    for step in range(training_steps):
        b_S, b_y, b_I = Agent.next_batch()
        Agent.learn(b_S,b_y,b_I)

        if (step % save_step ==0) and (step!=0) :
            loss_train = Agent.loss( Agent.memory['S'],
                                Agent.memory['y'],
                                Agent.memory['I'])

            loss_test = Agent.loss( Agent.memory_val['S'],
                                Agent.memory_val['y'],
                                Agent.memory_val['I'])
            if print_process:
                print('step:',step,
                    'trainning loss:', loss_train,
                    'testing loss:', loss_test)

            if len(LOSS_test)>1:
                if LOSS_test[-1] < loss_test:
                    early_stop += 1
                    if (early_stop > early_stop_number) and (step > int(training_steps*0.4)):
                        print('step:',step,
                            'trainning loss:', loss_train,
                            'testing loss:', loss_test)
                        
                        print('early stop')
                        break
                else:
                    early_stop = 0

            LOSS_train.append(loss_train)
            LOSS_test.append(loss_test)

    return Agent, LOSS_train, LOSS_test
