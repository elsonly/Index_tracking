from agent import Agent
from config import *

NNAgent = Agent(config)
LOSS = []

print('training')
for _ in range(10000):
	b_S, b_y, b_I = NNAgent.next_batch()
	NNAgent.learn(b_S,b_y,b_I)

	if (_ % 1 ==0) and (_!=0) :
		loss = NNAgent.loss(b_S, b_y, b_I)
		print(loss)
		LOSS.append(loss)



for i_episode in range(3000):

    observation = env.reset()

    action = RL.choose_action(observation)
    for day in range(days): #days = 128
        

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)    
  
    vt = RL.learn()























