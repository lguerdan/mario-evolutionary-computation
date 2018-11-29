from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import gym_super_mario_bros.actions 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np
import cPickle as pickle
from abc import ABCMeta, abstractmethod


#MSBGeneticOptimizerEnv contains boilerplate code for the project, shouldn't need to modify this
#If you do let me know or make a PR and I can fix it so that others have updated code too
class MSBGeneticOptimizerEnv(object):
	"""An environment wrapper for genetically optimizing mario smash brothers simulation ."""

	def __init__(self, max_steps=10000, num_genes=4, action_encoding=SIMPLE_MOVEMENT, render=False, reward="score", session_file=""):
		if session_file != "":
			self.load_optimizer(session_file)
		else: 
			self.max_steps = max_steps
			self.action_encoding = action_encoding
			self.render = render
			self.reward = reward  #score, x_pos, time, coins
			self.num_genes = num_genes
			self.init_chromosomes()

		self.init_simulator_env()

	def init_simulator_env(self):
		env = gym_super_mario_bros.make('SuperMarioBros-v0')
		self.env = BinarySpaceToDiscreteSpaceEnv(env, self.action_encoding)

	def init_chromosomes(self):
		"""Creates a new set of genes based on the number of parents fed in"""
		self.chromosomes = []
		for i in xrange(self.num_genes):
			chromosome = [np.random.randint(0,len(self.action_encoding), self.max_steps), -1]
			self.chromosomes.append(chromosome)

	def save_optimizer(self, fname):
		print "saving optimizer state to {}\n\n".format(fname)
		optimizer_state = Optimizer(self.max_steps, self.num_genes, self.action_encoding, self.render, self.reward, self.chromosomes)
		with open(fname, "wb") as f:
			pickle.dump(optimizer_state, f)

	def load_optimizer(self, fname):
		print "loading optimier state from {}\n\n".format(fname)
		with open(fname, "rb") as f:
			optimizer = pickle.load(f)
			self.max_steps = optimizer.max_steps
			self.action_encoding = optimizer.action_encoding
			self.render = optimizer.render
			self.reward = optimizer.reward
			self.num_genes = optimizer.num_genes
			self.chromosomes = optimizer.chromosomes

	def close_simulator_env(self):
		self.env.close()

	def run_generations(self, ngens):

		for gen in xrange(ngens):
			self.evaluate_genes()
			self.perform_selection()
			max_reward, max_reward_ix = self.get_max_reward_chromosome()
			print "\n#################################"
			print "GENERATION {} COMPLETE".format(gen)
			print "Highest chromosome: {}, reward: {}".format(max_reward_ix, max_reward)
			print "####################################\n\n\n"

	def get_max_reward_chromosome(self):
		"""returns highest reward of current chromosomes, along with its index"""
		max_reward = -1
		max_reward_ix = -1
		max_chromosome = []
		for cix, chromosome in enumerate(self.chromosomes):
			if chromosome[1] > max_reward:
				max_reward = chromosome[1]
				max_reward_ix = cix

		return max_reward, max_reward_ix

	@abstractmethod
	def perform_selection(self):
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
		max_reward, max_reward_ix = self.get_max_reward_chromosome()

		if max_reward == -1:
			print('Run top chromosome error: no rewards have been computed')

		done = True
		for step, action in enumerate(self.chromosomes[max_reward_ix][0]):
				if done:
					state = self.env.reset()

				state, reward, done, info = self.env.step(action)

				if render: self.env.render()


	def evaluate_genes(self):
		"""
		Given a gene structure, evaluates all genes for their reward and stores it in the stucture
		This is what actually runs the training simulation
		Input: a gene structure with (possibly) empty rewards
		Output: a gene structure with computed rewards
		"""

		for chromosome_num, chromosome in enumerate(self.chromosomes):
			done = True
			
			#Main evaluation loop for this chromosome
			for step, action in enumerate(chromosome[0]):
				
				if done: state = self.env.reset()

				state, reward, done, info = self.env.step(action)
				if step % 50 == 0:
					print('chromosome: {}, step: {}, action: {}, info: {}'.format(chromosome_num, step, action, info))
				
				if self.render: self.env.render()

			self.chromosomes[chromosome_num][1] = info[self.reward]
			print('chromosome {} done. {} reward = {}\n'.format(chromosome_num, self.reward, info[self.reward]))


	def __del__(self):
		"""close out of AI gym environment"""
		self.close_simulator_env()


class Optimizer():
	"""A basic container class for saving optimizer contents"""

	def __init__(self, max_steps, num_genes, action_encoding, render, reward, chromosomes):
		self.max_steps = max_steps
		self.num_genes = num_genes
		self.action_encoding = action_encoding
		self.render = render
		self.reward = reward
		self.chromosomes = chromosomes
