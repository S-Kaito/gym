import copy
import gym
from collections import deque
import keras
from keras import optimizers
from keras import losses
from keras.models import Sequential
from keras.models import model_from_config
from keras.models import model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf
import time

EPOCH = 1000
BATCH_SIZE = 32
TURN = 150
GAMMA = 0.95
	
def huberloss(y_true, y_pred):
    return K.mean(K.minimum(0.5*K.square(y_pred-y_true), K.abs(y_pred-y_true)-0.5), axis=1)

def loss_func(y_true, y_pred):
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    return loss

def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

def getAction(model,observation,episode):
	y = model.predict(observation.reshape((1,2)))
	if (0.01 +0.9/(1.0 + episode)) <= np.random.uniform(0,1):
		return -1 if np.argmax(y) == 0 else 1 , y
	else:
		return np.random.choice([-1, 1]) , y

def learn(model,target,data,height):
	y_pred = []
	y_true = []
	for d in data:
		state,action,reward,nextState = d
		y = model.predict(state)
		if height[0] != state[0][0] or not (nextState == np.zeros(state.shape)).all(axis=1):
			y[0][action] = reward + GAMMA * np.max(target.predict(nextState)[0])
		else:
			y[0][action] = 1 if height[0] > height[1] else -1
		y_pred.append(state.reshape(2))
		y_true.append(y.reshape(2))
	model.fit(np.array(y_pred),np.array(y_true),batch_size=BATCH_SIZE,verbose=0,epochs=1)

def main(args):

	env = gym.make('MountainCarContinuous-v0')
	
	if "-r" in args:
		model = model_from_json(open('model.json').read())
		model.load_weights('model.h5')
	else:
		model = Sequential()
		model.add(Dense(16,activation="relu",input_dim=2))
		model.add(Dense(16,activation="relu"))
		model.add(Dense(2,activation="linear"))
	target = clone_model(model)
	data = deque(maxlen=200)

	model.compile(loss=loss_func, optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0))
	model.summary()
		
	plot_x = []
	plot_y = []
	height = -1
	maxHeight = -1

	for episode in range(EPOCH):
		observation = env.reset()
		nextObservation, reward, done, info = env.step(env.action_space.sample())
		height = -1
		done = None

		for t in range(TURN):

			if EPOCH - episode == 2:
				env.render()
				time.sleep(0.1)

			action , y = getAction(model,observation,episode)
			nextObservation, reward, done, info = env.step([action])
	
			observation = nextObservation
			height = max(height,nextObservation[0])
			data.append((observation.reshape((1,2)),action,0,nextObservation.reshape((1,2))))

		print("{} times : finished after {} position {}".format(episode,height,maxHeight))

		if height < 0.45:
			data.append((observation.reshape((1,2)),action,-1,np.zeros((1,2))))
		else:
			data.append((observation.reshape((1,2)),action,1,np.zeros((1,2))))	

		plot_x.append(episode)
		plot_y.append(nextObservation[0])

		if BATCH_SIZE < len(data):
			learn(model,target,data,(height,maxHeight))
		if episode % 5 == 0:
			target = clone_model(model)

		maxHeight = max(maxHeight,height)

	plt.scatter(plot_x,plot_y,marker="+")
	plt.show()

main(sys.argv)