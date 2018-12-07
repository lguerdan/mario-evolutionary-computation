#!/usr/local/lib/python3.7

#import the optimizer framework
from evolution_env import EvolutionEnv


#create a new optimizer environment and initialize data structure
optimizer = EvolutionEnv(max_steps = 1000, num_chromosomes=8)

#run the optimizer for the desired number of times
optimizer.run_generations(1)

#serialize the data structure to a file. Pickle for effeciency 
# optimizer.save_optimizer('mario-4-chromosome.p')

#see how the top performing chromosome looks in the simulator 
# optimizer.run_top_chromosome(render=True)


#To load a previous environment, either create a new optimizer with the filename,
#or call the load optimizer function directly. Note this will overwrite the current state

#load the previous session and keep training
# optimizer2 = EvolutionEnv(session_file ="mario-4-chromosome.p")
# optimizer2.run_generations(1)
# optimizer2.save_optimizer('mario-4-chromosome2.p') #save to an updated file 