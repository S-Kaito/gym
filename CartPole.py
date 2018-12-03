import gym
import numpy as np
import random
import sys
import TensowFlow as tf

EPOCH = 100
BATCH_SIZE = 20
CLEAR_TURN = 50
GAMMA = 0.99

env = gym.make('CartPole-v0')
network = tf.Network([tf.LayerLeakyReLU(4,15),tf.LayerIdentity(15,2)])
data = []

def getAction(observation,episode):
	y = network.forward(observation)
	if (0.01 +0.9/(1.0 + episode)) <= np.random.uniform(0,1):
		return np.argmax(y)
	else:
		return np.random.choice([0, 1])

def learn():
	global network,data
	newNetwork = network.copy()
	random.shuffle(data)
	for d in data:
		state,action,reward,nextState = d
		y = np.copy(network.forward(state))
		y[0][action] = reward + GAMMA * np.max(network.forward(nextState)[0])
		newNetwork.forward(state)
		newNetwork.backward(y)
		newNetwork.update()
	network = newNetwork
	data = []

def main(args):
	avg = 0
	for episode in range(EPOCH):
		observation = env.reset()
		for t in range(CLEAR_TURN * 2):
			action = getAction(observation,episode)
			nextObservation, reward, done, info = env.step(action)
			print(action)
			if done:
				if t > CLEAR_TURN:
					data.append((observation.reshape((1,4)),action,1,np.zeros((1,4))))
				else:
					data.append((observation.reshape((1,4)),action,-1,np.zeros((1,4))))
				print("finished after {} timestamps".format(t+1))
				avg += 1
				break
			else:
				data.append((observation.reshape((1,4)),action,0,nextObservation.reshape((1,4))))
			observation = nextObservation
		if BATCH_SIZE < len(data):
			print("Avg = {} ".format(len(data) / avg),len(data))
			avg = 0
			learn()

main(sys.argv)