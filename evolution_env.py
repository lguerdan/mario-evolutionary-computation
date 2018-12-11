from msb_genetic_optimizer_env import MSBGeneticOptimizerEnv
import numpy as np


##The below class inherits everything from MSBGeneticOptimizerEnv, all you will need to do is modi
class EvolutionEnv(MSBGeneticOptimizerEnv):
   # Arguments: max_steps=10000, num_chromosomes=4, action_encoding=SIMPLE_MOVEMENT, render=False, reward="score", session_file=""
   def __init__(self, *args, **kwargs):
      super(EvolutionEnv, self).__init__(*args, **kwargs)

   def new_generation(self):
      """
      Based on a chromosomes structure, updates the chromosomes by natural selection rules
      This is where the bulk of the evolutionary computation code will go
      We will need to modify the chromosome structure in some
      """
        
      # elite selection for parent
      parents = self.select_parents(1, int(self.num_chromosomes/2))
      offspring = []

      # Crossover
      for i in range(int(np.floor(len(parents) / 2))):
         offspring.extend(self.crossover_chromosome_pair(parents[2 * i], parents[2 * i + 1],
                              point_num=2, points_before_death=True, normal_dist=True))

      # Mutation
      # using lambda = mu (numberof parents equal to the number of offsprings)
      # For each selected parent, will mutate around 20% of the max_steps actions
      mutations = round(0.2 * self.max_steps)
      for chromosome in parents:
         child = [chromosome[0].copy(), -1, chromosome[2]]
         self.mutate_chromosome(child, mutations)
         offspring.append(child)

      # (mu, lambda) Using only offspring to populate new generation here
      # self.chromosomes = offspring

      # (mu + lambda) Using parents + offspring to populate new generation here
      self.chromosomes = parents + offspring

   ###
   #  Basic implementation of parent selection
   #  - selection_type: 0: shuffle, 1: elite
   #  - mu: number of parents to select
   ###
   def select_parents(self, selection_type=0, mu=1):
      def shuffle_selection():
         np.random.shuffle(self.chromosomes)
         return self.chromosomes[:mu]

      def elite_selection(mu):
         parents = []
         # select mu best chromosomes
         best_chromosomes_index = sorted(range(len(self.chromosomes)), key=lambda x: self.chromosomes[x][1],
                                         reverse=True)
         # delete from chromosome list if it is not within the mu best
         for i in range(mu):
            parents.append(self.chromosomes[best_chromosomes_index[i]])
         return parents

      if selection_type == 0:
         return shuffle_selection()
      elif selection_type == 1:
         return elite_selection(mu)

   def crossover_chromosome_pair(self, parent1, parent2, point_num=1, points_before_death=False, normal_dist=False):
      ###
      #	Implementation of crossover functionality
      #	- points: number of points to use in crossover
      #	- points_before_death: only consider points upto sooner death
      #	- normal_dist: sample crossover points from normal distribution
      ###
      def positive_normal():
         x = abs(np.random.randn() / 3)
         return max(1 - x, 0)

      point_range_max = min(len(parent1[0]), len(parent2[0])) if not points_before_death \
         else max(parent1[2], parent2[2])
      random = np.random.rand if not normal_dist else positive_normal

      crossover_points = [point_range_max]
      for i in range(point_num):
         crossover_points.append(int(round(point_range_max * random())))
      crossover_points.sort(reverse=True)

      child1, child2 = [parent1[0].copy(), -1, -1], [parent2[0].copy(), -1, -1]
      while len(crossover_points) > 1:
         point1 = crossover_points.pop()
         point2 = crossover_points.pop()
         child1[0][point1:point2] = parent2[0][point1:point2].copy()
         child2[0][point1:point2] = parent1[0][point1:point2].copy()

      return [child1, child2]

   def mutate_chromosome(self, chromosome, mutations=1):
   ###
   # Implementation of mutation operator
   # Mutations: number of mutations to make
   ###
      for i in range(int(mutations)):
         #mutation index: triangular distribution with: bound left=0, bound right and mode=index of game over
         mutation_index = int(np.ceil(np.random.triangular(left=0, mode=1, right=1) * chromosome[2]))
         #new_action: random within the allowed possible actions
         actions_size = len(self.action_encoding)
         #new_action = np.random.randint(low=0, high=actions_size)
         #new: prioritize actions between 1 to 4 (going right)
         new_action = np.random.randint(low=0, high=(actions_size+4))
         if (new_action >= actions_size):
            new_action = new_action - actions_size + 1
         chromosome[0][mutation_index] = new_action
      chromosome[1], chromosome[2] = -1, -1
