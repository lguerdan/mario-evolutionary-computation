from msb_genetic_optimizer_env import MSBGeneticOptimizerEnv


##The below class inherits everything from MSBGeneticOptimizerEnv, all you will need to do is modi
class SelectionEnv(MSBGeneticOptimizerEnv):
	# Arguments: max_steps=10000, num_genes=4, action_encoding=SIMPLE_MOVEMENT, render=False, reward="score", session_file=""
	def __init__(self, *args, **kwargs):
		super(SelectionEnv, self).__init__(*args, **kwargs)

	def perform_selection(self):
		"""
		Based on a chromosomes structure, updates the chromosomes by natural selection rules
		This is where the bulk of the evolutionary computation code will go
		We will need to modify the chromosome structure in some 
		"""
		#

		pass 
		# For now now selection occurs, just keep current chromosomes
