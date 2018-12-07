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
        #
        parents = self.select_parents()
        offspring = []

        for (parent1, parent2) in parents:
            offspring.extend(self.crossover_chromosome_pair(parent1, parent2))

        for chromosome in offspring:
            self.mutate_chromosome(chromosome, mutations=5)

        # Using only offspring to populate new generation here
        self.chromosomes = offspring

    ###
    #	Basic implementation of parent selection, with chromosomes shuffled then returned in pairs
    ###
    def select_parents(self):
        np.random.shuffle(self.chromosomes)
        parents = []
        for i in range(int(len(self.chromosomes) / 2)):
            parents.append((self.chromosomes[2 * i], self.chromosomes[2 * i + 1]))

        return parents

    ###
    #	Implementation of crossover functionality
    #	- points: number of points to use in crossover
    #	- points_before_death: only consider points upto sooner death
    #	- normal_dist: sample crossover points from normal distribution
    ###
    def crossover_chromosome_pair(self, parent1, parent2, point_num=1, points_before_death=False, normal_dist=False):
        def positive_normal():
            x = (np.random.randn() / 3)
            return max(1 - max(x, -x), 0)

        point_range_max = min(len(parent1[0]), len(parent2[0])) if not points_before_death \
            else max(parent1[2], parent2[2])
        random = np.random.rand if not normal_dist else positive_normal

        crossover_points = [point_range_max]
        for i in range(point_num):
            crossover_points.append(round(point_range_max * random()))
        crossover_points.sort(reverse=True)

        child1, child2 = [parent1[0].copy(), -1, -1], [parent2[0].copy(), -1, -1]
        while len(crossover_points) > 1:
            point1 = crossover_points.pop()
            point2 = crossover_points.pop()
            child1[0][point1:point2] = parent2[0][point1:point2].copy()
            child2[0][point1:point2] = parent1[0][point1:point2].copy()

        return [child1, child2]

    ###
    #	Basic implementation of mutation operator
    #	- mutations: number of mutations to make, at randomly selected positions
    ###
    def mutate_chromosome(self, chromosome, mutations=1):
        for _ in range(mutations):
            mutation_point = 0  # round(np.random.rand() * len(chromosome[0]))
            mutation = round(np.random.rand() * len(self.action_encoding))
            chromosome[0][mutation_point] = mutation
