import gymnasium


"""
	This class implements a wrapper for a standard gymnasium environment, it has two main purposes:

		> Fix the environments with only one cost function, returned as float in the info dictionary, in this
		  case we force the standard form for our approach, i.e., an array of one element for each cost function;
		  (e.g., info['cost]=0 => info['cost]=[0]).
		> It also implements a monitor to avoid problems with the normalization of the cost and reward functions, 
	      it takes track of the non-normalized cost function, useful for the computation of the lambda loss but 
		  also for performance monitoring (the real value is stored in the 'cost_monitor' list).
"""
class MultiCostWrapper( gymnasium.Wrapper ):
	
	
	"""
		Initialization of the wrapper, basically it is a copy of the standard environment 
		where we overrite the basic function (e.g., reset and step) to perform additional
		operations.
	"""
	def __init__ ( self, env ):
		super().__init__(env)
		self.reward_monitor = None
		self.cost_monitor = None


	"""
		Override the 'reset' function to reset internal monitoring variables; notice that we must return
		the same value returned by the original 'reset' function.
	"""
	def reset( self, **kwargs ):

		# First recall the 'true' reset function of the environment
		obs, info = self.env.reset( **kwargs )

		# Reset of the monitoring variables and return the 'true' results
		self.reward_monitor = 0
		self.cost_monitor = None
		return obs, info
		

	"""
		Override the 'step' function to update the internal monitors; this method performs also
		the 'single cost' fix (i.e., info['cost]=0 => info['cost]=[0]).
	"""
	def step( self, action):

		# First recall the 'true' step function of the environment
		# storing the results
		obs, reward, terminated, truncated, info = self.env.step(action)

		# Fix for single cost and initialization of the cost-monitor; this init
		# is performed only once after each reset to match the number of cost functions
		# of the enviornment. Fix also for the environments without 
		# the cost (i.e., classical gymnasium like 'CartPole')
		if not "cost" in info.keys(): info["cost"] = [0.0]
		if type(info["cost"]) is not list: info["cost"] = [info["cost"]]
		if self.cost_monitor is None: self.cost_monitor = [0.0 for _ in info["cost"]]

		# Update of cost and reward monitors; notice that this operation must be perfomed
		# before the normalization steps!
		self.reward_monitor += reward
		for idx, c in enumerate(info["cost"]): self.cost_monitor[idx] += c

		# In the last episode, we returned the results of the monitor as part of 
		# the info dictionary
		info['custom_info'] = {
			'tot_reward': self.reward_monitor,
			'tot_costs': self.cost_monitor
		}

		#
		return obs, reward, terminated, truncated, info
	
