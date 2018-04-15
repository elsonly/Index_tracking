import tensorflow as tf

from agent import Agent
from config import *

tf.reset_default_graph()

NNAgent = Agent(config)
LOSS_train = []
LOSS_test = []

print('start trainning ...')
for step in range(10000):
	b_S, b_y, b_I = NNAgent.next_batch()
	NNAgent.learn(b_S,b_y,b_I)

	if (step % 500 ==0) and (step!=0) :
		loss_train = NNAgent.loss( NNAgent.memory['S'],
							NNAgent.memory['y'],
							NNAgent.memory['I'])

		loss_test = NNAgent.loss( NNAgent.memory_test['S'],
							NNAgent.memory_test['y'],
							NNAgent.memory_test['I'])
		print('step:',step,
			'trainning loss:', loss_train,
			'testing loss:', loss_test)
		LOSS_train.append(loss_train)
		LOSS_test.append(loss_test)


"""
for i_episode in range(3000):

    observation = env.reset()

    action = RL.choose_action(observation)
    for day in range(days): #days = 128
        

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)    
  
    vt = RL.learn()

"""