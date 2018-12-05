import gym
from collections import deque
import keras
from keras import optimizers
from keras import losses
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time

EPOCH = 200
BATCH_SIZE = 32
CLEAR_TURN = 100
GAMMA = 0.99

def getAction(model,observation,episode):
	y = model.predict(observation.reshape((1,4)))
	if (0.01 +0.9/(1.0 + episode)) <= np.random.uniform(0,1):
		return np.argmax(y)
	else:
		return np.random.choice([0, 1])

def learn(model,data):
	y_pred = []
	y_true = []
	for d in data:
		state,action,reward,nextState = d
		y = model.predict(state)
		if not (nextState == np.zeros(state.shape)).all(axis=1):
			y[0][action] = reward + GAMMA * np.max(model.predict(nextState)[0])
		else:
			y[0][action] = reward
		y_pred.append(state.reshape(4))
		y_true.append(y.reshape(2))
	model.fit(np.array(y_pred),np.array(y_true),batch_size=BATCH_SIZE,verbose=0,epochs=1)
	
def huberloss(y_true, y_pred):
    return K.mean(K.minimum(0.5*K.square(y_pred-y_true), K.abs(y_pred-y_true)-0.5), axis=1)

def main(args):

	env = gym.make('CartPole-v0')
	model = Sequential()
	model.add(Dense(16,activation="relu",input_dim=4))
	model.add(Dense(16,activation="relu"))
	model.add(Dense(2,activation="linear"))
	model.compile(loss=huberloss, optimizer=Adam(lr=0.00001))
	model.summary()

	data = deque(maxlen=200)

	plot_x = []
	plot_y = []

	point = 0

	for episode in range(EPOCH):
		observation = env.reset()
		nextObservation, reward, done, info = env.step(env.action_space.sample())

		for t in range(CLEAR_TURN * 2):

			if EPOCH - episode == 2:
				env.render()
				time.sleep(0.1)

			action = getAction(model,observation,episode)
			nextObservation, reward, done, info = env.step(action)
			if done:
				if t > CLEAR_TURN:
					data.append((observation.reshape((1,4)),action,1,np.zeros((1,4))))
				else:
					data.append((observation.reshape((1,4)),action,-1,np.zeros((1,4))))
				print("{} times : finished after {} timestamps".format(episode,t+1))
	
				plot_x.append(episode)
				plot_y.append(t + 1)

				break
			else:
				data.append((observation.reshape((1,4)),action,0,nextObservation.reshape((1,4))))
			observation = nextObservation
		if BATCH_SIZE < len(data):
			learn(model,data)

	plt.scatter(plot_x,plot_y,marker="+")
	plt.show()

main(sys.argv)