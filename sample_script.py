
#import the optimizer framework
from msb_genetic_optimizer import MSBGeneticOptimizerEnv

#create a new optimizer environment
optimizer = MSBGeneticOptimizerEnv(max_steps=50, reward='time')

#create a new genome for our session
optimizer.new_optimizer_session(4,'silly')
# optimizer.evaluate_genes()
optimizer.evaluate_genes()


