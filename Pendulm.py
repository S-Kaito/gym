import gym


env = gym.make('Pendulum-v0')  # make your environment!

population = [[[3] for _ in range(40)] for _ in range(100)]

for i_episode in range(20):
	observation = env.reset()

	for t in range(100):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		print(t, observation)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
	print(observation.shape)