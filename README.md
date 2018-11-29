First, see the [simulator module](https://github.com/lguerdan/gym-super-mario-bros) for information about how to configure the python environment. 

# Info about the wrapper

## Chromosomes:

An python list of lists, where each list entry corresponds to a chromosome and its award

The initial numpy array corresponds to the chromosome: 
It will be max_steps long and each value will be in the range: 0 -> max step
Not yet evaluated chromosomes have an award of -1
```
[
    [[1,2,0,3,2,....], -1]
    [[0,6,1,3,2,....], -1]
]
```
Each step in a chromosome encodes an action in the simulator

## SelectionEnv
Only need to update `perform_selection` based on each of our methods. Can overload function definition etc. 


## MSBGeneticOptimizerEnv
Contains bulk of the code. Important functions: 
- `run_generations`: Runs main loop of all chromosomes for ngens times 
- `run_top_chromosome`: Used to visualize how the best solution is running. Test phase more or less
- `evaluate_genes`: Runs each chromosome through the simulator and stores the corresponding reward
- `save_optimizer`: Saves the current optimizer state to a file
- `load optimizer`: Loads the optimizer with previously computed state for continued training or visualization of good solutions

## Usage
create a new optimizer environment and initialize data structure
`optimizer = SelectionEnv(max_steps = 500, num_genes=6, render=True)`

run the optimizer for the desired number of generations
`optimizer.run_generations(3)`

serialize the data structure to a file. Pickle for effeciency 
`optimizer.save_optimizer('mario-4-chromosome.p')`

#see how the top performing chromosome looks in the simulator 
`optimizer.run_top_chromosome(render=True)`


To load a previous environment, either create a new optimizer with the filename,or call the load optimizer function directly. Note this will overwrite the current state

`optimizer2 = SelectionEnv(session_file ="mario-4-chromosome.p")`
load the previous session and keep training
`optimizer2.run_generations(1)`
`optimizer2.save_optimizer('mario-4-chromosome2.p') #save to an updated file `
