First, see the (simulator module)[gym-super-mario-bros/README.md] for information about how to configure the python environment. 

#Info about the wrapper

## Chromosomes:

An python list of lists, where each list entry corresponds to a chromosome and its award

The initial numpy array corresponds to the chromosome: 
It will be max_steps long and each value will be in the range: 0 -> max step
Not yet evaluated chromosomes have an award of -1
[
    [[1,2,0,3,2,....], -1]
    [[0,6,1,3,2,....], -1]
]

Each step in a chromosome encodes an action in the simulator

##SelectionEnv
Only need to update `perform_selection` based on each of our methods. Can overload function definition etc. 


##MSBGeneticOptimizerEnv
Contains bulk of the code. Important functions: 
- `run_generations`: Runs main loop of all chromosomes for ngens times 
- `run_top_chromosome`: Used to visualize how the best solution is running. Test phase more or less
- `evaluate_genes`: Runs each chromosome through the simulator and stores the corresponding reward
- `save_optimizer`: Saves the current optimizer state to a file
- `load optimizer`: Loads the optimizer with previously computed state for continued training or visualization of good solutions