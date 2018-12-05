from __future__ import print_function
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import gym_super_mario_bros.actions 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from abc import ABCMeta, abstractmethod


#MSBGeneticOptimizerEnv contains boilerplate code for the project, shouldn't need to modify this
#If you do let me know or make a PR and I can fix it so that others have updated code too
class MSBGeneticOptimizerEnv(object):
	"""An environment wrapper for genetically optimizing mario smash brothers simulation ."""

	def __init__(self, max_steps=10000, num_chromosomes=4, action_encoding=SIMPLE_MOVEMENT, render=False, fitness_strategy="score", session_file=""):
		if session_file != "":
			self.load_optimizer(session_file)
		else: 
			self.max_steps = max_steps
			self.action_encoding = action_encoding
			self.render = render
			self.fitness_strategy = fitness_strategy  #score, x_pos, time, coins
			self.num_chromosomes = num_chromosomes
			self.init_chromosomes()

		self.init_simulator_env()

	def init_simulator_env(self):
		env = gym_super_mario_bros.make('SuperMarioBros-v0')
		self.env = BinarySpaceToDiscreteSpaceEnv(env, self.action_encoding)

	def init_chromosomes(self):
		"""Creates a new set of genes based on the number of parents fed in"""
		self.chromosomes = []
		for i in range(self.num_chromosomes):
			chromosome = [np.random.randint(0,len(self.action_encoding), self.max_steps), -1, -1]
			self.chromosomes.append(chromosome)

	def save_optimizer(self, fname):
		print("saving optimizer state to ",fname)
		optimizer_state = Optimizer(self.max_steps, self.num_chromosomes, self.action_encoding, self.render, self.fitness_strategy, self.chromosomes)
		with open(fname, "wb") as f:
			pickle.dump(optimizer_state, f)

	def load_optimizer(self, fname):
		print("loading optimier state from ",fname)
		with open(fname, "rb") as f:
			optimizer = pickle.load(f)
			self.max_steps = optimizer.max_steps
			self.action_encoding = optimizer.action_encoding
			self.render = optimizer.render
			self.fitness_strategy = optimizer.fitness_strategy
			self.num_chromosomes = optimizer.num_chromosomes
			self.chromosomes = optimizer.chromosomes

	def close_simulator_env(self):
		self.env.close()

	def run_generations(self, ngens):

		for gen in range(ngens):
			self.evaluate_chromosomes()
			self.new_generation()
			max_fitness, max_fitness_ix = self.get_max_fitness_chromosome()
			print("\n#################################")
			print("GENERATION",gen,"COMPLETE")
			print("Highest chromosome: ",max_fitness_ix,", fitness:",max_fitness)
			print("####################################\n\n\n")

	def get_max_fitness_chromosome(self):
		"""returns highest fitness of current chromosomes, along with its index"""
		max_fitness = -1
		max_fitness_ix = -1
		max_chromosome = []
		for cix, chromosome in enumerate(self.chromosomes):
			if chromosome[1] > max_fitness:
				max_fitness = chromosome[1]
				max_fitness_ix = cix

		return max_fitness, max_fitness_ix

	@abstractmethod
	def new_generation(self):
		"""
		Based on a chromosomes structure, updates the chromosomes by natural selection rules
		This is where the bulk of the evolutionary computation code will go
		Update here
		"""
		#
		pass 
		# For now now selection occurs, just keep current chromosomes


	def run_top_chromosome(self, render=False):
		"""
		Retrieve the best-performing chromosome and play it. 
		Override render argument in case only want to visualize on test round
		"""
		max_fitness, max_fitness_ix = self.get_max_fitness_chromosome()

		if max_fitness == -1:
			print('Run top chromosome error: no fitnesses have been computed')

		done = True
		for step, action in enumerate(self.chromosomes[max_fitness_ix][0]):
				if done:
					state = self.env.reset()

				state, reward, done, info = self.env.step(action)

				if render: self.env.render()


	def evaluate_chromosomes(self):
		"""
		Given a gene structure, evaluates all genes for their fitness and stores it in the stucture
		This is what actually runs the training simulation
		Input: a gene structure with (possibly) empty fitnesses
		Output: a gene structure with computed fitnesses and index of death
		"""

		for chromosome_num, chromosome in enumerate(self.chromosomes):
			done = True
			step_died = -1
			#Main evaluation loop for this chromosome
			for step, action in enumerate(chromosome[0]):
				
				#rest new environment
				if done: 
					state = self.env.reset()

				#take step
				state, reward, done, info = self.env.step(action)
				
				#died
				if info['life'] < 3:
					step_died = step
					break

				#print progress
				if step % 50 == 0:
					print("chromosome:",chromosome_num," step:", step," action:",action, "info:",info)
				
				#display on screen
				if self.render: 
					self.env.render()

			self.chromosomes[chromosome_num][1] = info[self.fitness_strategy]
			self.chromosomes[chromosome_num][2] = step
			print("chromosome ",chromosome_num," done fitness ",self.fitness_strategy ,"= ",info[self.fitness_strategy])


	def __del__(self):
		"""close out of AI gym environment"""
		self.close_simulator_env()


class Optimizer():
	"""A basic container class for saving optimizer contents"""

	def __init__(self, max_steps, num_chromosomes, action_encoding, render, fitness_strategy, chromosomes):
		self.max_steps = max_steps
		self.num_chromosomes = num_chromosomes
		self.action_encoding = action_encoding
		self.render = render
		self.fitness_strategy = fitness_strategy
		self.chromosomes = chromosomes
