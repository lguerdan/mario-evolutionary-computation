from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import gym_super_mario_bros.actions 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np

class MSBGeneticOptimizerEnv():
	"""An environment wrapper for genetically optimizing mario smash brothers simulation ."""

	def __init__(self, max_steps=10000, action_encoding=SIMPLE_MOVEMENT, render=False, reward="score"):
		self.max_steps = max_steps
		self.action_encoding = action_encoding
		self.render = render
		self.reward = reward  #score, x_pos, time, coins
		self.init_env()


	def init_env(self):
		env = gym_super_mario_bros.make('SuperMarioBros-v0')
		self.env = BinarySpaceToDiscreteSpaceEnv(env, self.action_encoding)

	def close_env(self):
		self.env.close()

	def new_optimizer_session(self, num_genes, reward):
		"""Creates a new set of genes based on the number of parents fed in"""

		self.chromosomes = []
		for i in xrange(num_genes):
			chromosome = [np.random.randint(0,len(self.action_encoding), self.max_steps), -1]
			self.chromosomes.append(chromosome)


	def perform_selection(self):
		"""
		Based on a chromosomes structure, updates the chromosomes by natural selection rules
		This is where the bulk of the evolutionary computation code will go
		Update here
		"""


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
				if done:
					state = self.env.reset()

				state, reward, done, info = self.env.step(action)
				if step % 10 == 0:
					print('chromosome: {}, step: {}, action: {}, info: {}'.format(chromosome_num, step, action, info))

				if self.render: self.env.render()

			self.chromosomes[chromosome_num][1] = info[self.reward]
			print('chromosome {} done. {} reward = {}\n'.format(chromosome_num, self.reward, info[self.reward]))


	# def load_optimizer_session(self, session_fname):
	# 	if session_fname == "":
	# 		print("Creating new optimizer session")

	# def save_optimizer_session(self, session_fname):


