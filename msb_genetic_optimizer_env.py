from __future__ import print_function

import multiprocessing, time, functools, gym_super_mario_bros, os, csv

from multiprocessing import Pool
from contextlib import contextmanager
from itertools import product
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

try:
   import cPickle as pickle
except:
   import pickle


#MSBGeneticOptimizerEnv contains boilerplate code for the project, shouldn't need to modify this
#If you do let me know or make a PR and I can fix it so that others have updated code too
class MSBGeneticOptimizerEnv(object):
   """An environment wrapper for genetically optimizing mario smash brothers simulation ."""

   def __init__(self, max_steps=3000, num_chromosomes=40, action_encoding=RIGHT_ONLY, render=False, fitness_strategy="x_pos", session_file="", world=1, stage=1, version=0, noFrameSkip=False):
      if session_file != "":
         self.load_optimizer(session_file)
      else:
         self.max_steps = max_steps
         self.action_encoding = action_encoding
         self.render = render
         self.fitness_strategy = fitness_strategy  #score, x_pos, time, coins
         self.num_chromosomes = num_chromosomes
         self.world = world
         self.stage = stage
         self.version = version
         self.noFrameSkip = 'NoFrameskip' if noFrameSkip else ''
         self.init_chromosomes()

   def init_chromosomes(self):
      """Creates a new set of genes based on the number of parents fed in"""
      self.chromosomes = []
      for i in range(self.num_chromosomes):
         chromosome = [np.random.randint(0,len(self.action_encoding), self.max_steps), -1, -1]
         self.chromosomes.append(chromosome)
      self.evaluate_chromosomes()

   def save_optimizer(self, fname):
      print("saving optimizer state to ",fname)
      optimizer_state = Optimizer(self.max_steps, self.num_chromosomes, self.action_encoding, self.render, self.fitness_strategy, self.chromosomes, self.world, self.stage, self.version, self.noFrameSkip)
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
         self.world = optimizer.world
         self.stage = optimizer.stage
         self.version = optimizer.version
         self.noFrameSkip = optimizer.noFrameSkip


   def run_generations(self, ngens, fname):

      headers = ['generation', 'chromosome_num', self.fitness_strategy, 'avg_fitness']

      #If logging for the first time, set up csv file
      if fname:
         print("logging progress to " + fname)

         with open (fname, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
            writer.writeheader()

      for gen in range(ngens):
         self.new_generation()
         self.evaluate_chromosomes()
         max_fitness, max_fitness_ix = self.get_max_fitness_chromosome()
         avg_fitness = self.get_avg_chromosome_fitness()

         #If writing progress to output, add this generation
         if fname:
            with open (fname, 'a') as csvfile:
               writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
               writer.writerow({'generation': gen, 'chromosome_num': max_fitness_ix, self.fitness_strategy: max_fitness, 'avg_fitness': avg_fitness})

         print("\n#################################")
         print("GENERATION",gen,"COMPLETE")
         print("Highest chromosome: ",max_fitness_ix,", fitness:",max_fitness)
         print("Average fitness: ", avg_fitness)
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

   def get_avg_chromosome_fitness(self):
      total_fitness = 0
      for chromosome in self.chromosomes:
         total_fitness += chromosome[1]

      return int(total_fitness / len(self.chromosomes))

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

      with mariocontext(self) as env:
         done = True
         for step, action in enumerate(self.chromosomes[max_fitness_ix][0]):

            state, reward, done, info = env.step(action)
            if done or or info['flag_get']:
               state = env.reset()

            if render: env.render()


   def evaluate_chromosome(self, input_tuple):
      """Evaluates a chromosome for it's fitness value and index of death"""

      chromosome_num, chromosome = input_tuple

      if chromosome[1] != -1:
         return chromosome

      with mariocontext(self) as env:

         best_fitness_step = 0

         state = env.reset()
         #Main evaluation loop for this chromosome
         for step, action in enumerate(chromosome[0]):

            #take step
            state, reward, done, info = env.step(action)

            if (info[self.fitness_strategy] > best_fitness_step):
               best_fitness_step = step

            #died or level beat
            if (done or info['flag_get']):
               break

            #print progress
            if step % 50 == 0:
               print("chromosome:",chromosome_num," step:", step," action:",action, "info:",info)

            #display on screen
            if self.render:
               env.render()

         chromosome[1], chromosome[2] = info[self.fitness_strategy], best_fitness_step

         print("chromosome",chromosome_num," done fitness ",self.fitness_strategy ,"= ",info[self.fitness_strategy])
         return chromosome


   def evaluate_chromosomes(self):
      """
      Given a gene structure, evaluates all genes for their fitness and stores it in the stucture
      This is what actually runs the training simulation
      Input: a gene structure with (possibly) empty fitnesses
      Output: a gene structure with computed fitnesses and index of death
      """

      bound_instance_method_alias = functools.partial(_instance_method_alias, self)
      with poolcontext(multiprocessing.cpu_count()) as pool:
         self.chromosomes = pool.map(bound_instance_method_alias, enumerate(self.chromosomes))


class Optimizer():
   """A basic container class for saving optimizer contents"""

   def __init__(self, max_steps, num_chromosomes, action_encoding, render, fitness_strategy, chromosomes, world, stage, version, noFrameSkip):
      self.max_steps = max_steps
      self.num_chromosomes = num_chromosomes
      self.action_encoding = action_encoding
      self.render = render
      self.fitness_strategy = fitness_strategy
      self.chromosomes = chromosomes
      self.world = world
      self.stage = stage
      self.version = version
      self.noFrameSkip = noFrameSkip


@contextmanager
def poolcontext(*args, **kwargs):
   pool = multiprocessing.Pool(*args, **kwargs)
   yield pool
   pool.terminate()

@contextmanager
def mariocontext(marioEnv):
   mario_env = 'SuperMarioBros' + marioEnv.noFrameSkip + '-' + str(marioEnv.world) + '-' + str(marioEnv.stage) + '-v' + str(marioEnv.version)
   env = gym_super_mario_bros.make(mario_env)
   env = BinarySpaceToDiscreteSpaceEnv(env, marioEnv.action_encoding)
   yield env
   env.close()


def _instance_method_alias(obj, arg):
   """
   Alias for instance method that allows the method to be called in a
   multiprocessing pool
   """
   return obj.evaluate_chromosome(arg)


