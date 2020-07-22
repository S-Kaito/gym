
from __future__ import print_function

import gym
import numpy as np
import itertools
import os
import neat
from neat import nn, population, statistics

np.set_printoptions(threshold=np.inf)
env = gym.make('CartPole-v1')

# run through the population


def eval_fitness(genomes, config):
	print(config)
	for g in genomes:
		observation = env.reset()
		# env.render()
		net = nn.create_feed_forward_phenotype(g)
		fitness = 0
		reward = 0
		frames = 0
		total_fitness = 0

		for k in range(5):
			while 1:
				inputs = observation

				# active neurons
				output = net.serial_activate(inputs)

				output = np.clip(output, -1, 1)
				# print(output)
				observation, reward, done, info = env.step(np.argmax(output))


				fitness += 1
				frames += 1
				# env.render()
				if done or frames > 2000:
					total_fitness += fitness
					# print(fitness)
					observation = env.reset()
					break
		# evaluate the fitness
		g.fitness = total_fitness / 5
	print(max([genomes[i].fitness for i in range(len(genomes))]))

local_dir = os.path.dirname(__file__)
config_path = os.path.abspath(os.path.join(local_dir, "CartPole02Conf"))

pop = population.Population(neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,config_path))

pop.run(eval_fitness, 2000)
winner = pop.statistics.best_genome()
del pop

winningnet = nn.create_feed_forward_phenotype(winner)

streak = 0

while streak < 100:
	fitness = 0
	frames = 0
	reward = 0
	observation = env.reset()
	env.render()
	while 1:
		inputs = observation

		# active neurons
		output = winningnet.serial_activate(inputs)
		output = np.clip(output, -1, 1)
		# print(output)
		observation, reward, done, info = env.step(np.argmax(output))

		fitness += 1

		env.render()
		frames += 1

		if done or frames > 2000:
			print(fitness)
			print ('streak: ', streak)
			if fitness >= 170:
					streak += 1
			else:
				streak = 0

			break
print("completed!")