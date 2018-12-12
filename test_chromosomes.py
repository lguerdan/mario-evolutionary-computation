#!/usr/local/lib/python3.7

#import the optimizer framework
from evolution_env import EvolutionEnv


# # #create a new optimizer environment and initialize data structure
# optimizer = EvolutionEnv(max_steps = 1000, num_chromosomes=20)
# optimizer.run_generations(100, '20-chromosomes-100-generations.csv')
# optimizer.save_optimizer('20-chromosomes-100-generations.p')# #create a new optimizer environment and initialize data structure

# optimizer2 = EvolutionEnv(max_steps = 1000, num_chromosomes=40)
# optimizer2.run_generations(100, '40-chromosomes-100-generations.csv')
# optimizer2.save_optimizer('40-chromosomes-100-generations.p')

optimizer3 = EvolutionEnv(max_steps = 1000, num_chromosomes=60)
optimizer3.run_generations(100, '60-chromosomes-100-generations.csv')
optimizer3.save_optimizer('60-chromosomes-100-generations.p')

optimizer4 = EvolutionEnv(session_file='60-chromosomes-100-generations.p')
optimizer4.run_top_chromosome(render=True)


#To load a previous environment, either create a new optimizer with the filename,
#or call the load optimizer function directly. Note this will overwrite the current state

#load the previous session and keep training
# optimizer = EvolutionEnv(session_file ="mario-4-chromosome.p")
# optimizer.run_top_chromosome(render=True)
# optimizer2.run_generations(1, 'fitness-values2.csv')
# optimizer2.save_optimizer('mario-4-chromosome2.p') #save to an updated file 