import copy
import gym
from gym import wrappers
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time

EPOCH = 1000
env = gym.make('MountainCarContinuous-v0')
env = wrappers.Monitor(env,'./video',video_callable=(lambda ep: int(round((ep / 100) ** (1. / 3))) ** 3 == (ep / 100)),force=True)

def action(ind,show=False,rec=False):

	env.reset()
	observation, reward, done, info = env.step([0])
	height = -1
	turn = 0
	done = None
	for i in range(0,5):
		for t in ind:
			if show:
				env.render()
			if t == 0:
				observation, reward, done, info = env.step([1])
			else:
				observation, reward, done, info = env.step([-1])

			height = max(height,observation[0])
			
			if not done:
				turn += 1
			else:
				return turn,

def mutate(ind,indpb=0.1):

	
def main(args):
	
	plot_x = []
	plot_y = []
	height = -1
	maxHeight = -1

	creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()

	toolbox.register("attr_bool", random.randint, 0, 1)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 200)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	toolbox.register("evaluate", action)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", mutate, indpb=0.05)
	toolbox.register("select", tools.selTournament, tournsize=3)

	pop = toolbox.population(n=99)
	CXPB, MUTPB, NGEN = 0.5, 0.2, 40

	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	print("Start of evolution")

	for episode in range(EPOCH):

		offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

		for child1, child2 in zip(offspring[::2], offspring[1::2]):

			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:

			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values
	
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		
		pop[:] = offspring
		
		fits = [ind.fitness.values for ind in pop]
		
		print(episode,"|",min(fits))

		plot_x.append(episode)
		plot_y.append(min(fits))

		action(tools.selBest(pop, 1)[0])
	
	print("-- End of (successful) evolution --")
	
	best_ind = tools.selBest(pop, 1)[0]
	print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

	plt.scatter(plot_x,plot_y,marker="+")
	plt.show()

main(sys.argv)