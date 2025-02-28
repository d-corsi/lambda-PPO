from scripts.utils import PolicyNetwork, DiscretePolicyNetwork, ValueNetwork, MemoryBuffer
from scripts.wrappers import MultiCostWrapper
import gymnasium, torch, numpy, random
import collections, abc, wandb, os, sys, yaml
	

""" 
	Parent class that implements the standard function for a deep reinforcement
	learning algorithm. It manages also the logging feature (i.e., printing or WandB).

	The current implementation is designed to work with a 'Gymnasium' enviornment [1] and
	support multiple cost functions; the values of the costs function must be returned in
	the 'info' dictionary by the 'step' function of the environment (e.g., info['cost]=[0, 1, 0]).
	Multiple environments can be run in parellel exploiting the 'AsyncVectorEnv' wrapper 
	of Gymnasium. In addition a custom wrapper (MultiCostWrapper) automatically fix the 
	environments with only one cost function building an array of 1 element for consistency 
	(e.g., info['cost]=0 => info['cost]=[0]).

	[1] https://gymnasium.farama.org
"""
class ReinforcementLearning( abc.ABC ):
    

	"""
		Initialization of all the neural networks, lagrangian multipliers, random
		seeds, and additional variables.
	"""
	def __init__(self, args ):

		# Loading the 
		self.args = args
		self.current_update = 0
		
		# Setting the random seed, if no seed has been set generating 
		# a new random seed
		if self.args.seed is None: self.args.seed = numpy.random.randint(0, 1000)
		random.seed( self.args.seed )
		numpy.random.seed( self.args.seed )
		torch.manual_seed( self.args.seed )

		# The training has been extensively tested in 'cpu' mode, but should
		# work also in 'cuda' mode
		self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Generation of multiple environments for the vectorized approach. Multiple enviornment run
		# in parallel for a faster data collection. It also guarantee a larger variety of data collected
		self.envs = gymnasium.vector.AsyncVectorEnv( [self._make_env() for _ in range(self.args.num_envs) ] )

		# Initialize the neural networks for actor and critic (reward-wise); we implemented
		# a policy network for both discrete and continuous action spaces
		if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete): self.actor = DiscretePolicyNetwork( self.envs ).to(self.device)
		else: self.actor = PolicyNetwork( self.envs ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.learning_rate, eps=1e-5)
		self.critic = ValueNetwork( self.envs ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.learning_rate, eps=1e-5)
		
		# Initialize the neural networks for the critics (cost-wise); notice that we generate
		# multiple cost critics (one for each cost function) to support a multi-objective setting
		self.cost_critics = [ValueNetwork( self.envs ).to(self.device) for _ in range(self.args.num_costs)]
		self.cost_critic_optimizers = [torch.optim.Adam(self.cost_critics[i].parameters(), lr=self.args.learning_rate, eps=1e-5) for i in range(self.args.num_costs)]

		# Initialize the optimizers for the lagrangian multipliers (one for each cost function)	
		self.lagrangian_multipliers = [torch.nn.Parameter( torch.as_tensor(self.args.lambda_init), requires_grad=True ) for _ in range(self.args.num_costs)]
		self.lambda_optimizers = [torch.optim.Adam( [self.lagrangian_multipliers[i]], lr=self.args.lambda_learning_rate ) for i in range(self.args.num_costs)]
		

	"""
		Mehtod that implements the main deep reinforcement learning loop. It performs all the data-collection operations 
		from the environment, select the actions to perform and takes track of the results. If required it publish
		the collected results on WandB.
	"""
	def main_loop( self ):

		# Activate wandb to save the logging data (if required)
		if self.args.verbose > 0: self._activateWandB()

		# Create the memory buffer object for the training loop
		memory_buffer = MemoryBuffer(  self.args.num_steps, self.args.num_envs, self.envs, self.device, self.args.num_costs )

		# Initialize logging arrays and queue
		ep_costs = []
		ep_rewards = []
		ep_costs_queue = collections.deque(maxlen=100)
		ep_rewards_queue = collections.deque(maxlen=100)

		# Starting the training loop, iterated 'updates' times given as paramter for the loop
		for update in range(1, (self.args.updates+1)):

			# Reset the environment before starting the process, remember that to propagate
			# the state through the neural network we must first convert it into a tensor.
			# We also add the update step to the seed to avoid overfitting while keeping
			# the reproducibility
			state, _ = self.envs.reset( seed=(self.args.seed+update) )
			state = torch.Tensor(state).to(self.device)

			# This support variable is useful for the delayed 'start train lambda', it his inherited
			# by the 'lagrangianPPO' class that exploits it.
			self.current_update = update

			# Initialize logging arrays for the current training iteration, notice that
			# for logging purpose we consider only the first environment of the vector. If 
			# in a single update we have multiple episodes of the environment we save the average 
			# returns
			ep_costs.append([])
			ep_rewards.append([])

			# Memory buffer re-initialization; policy gradient methods require the reset after each policy update
			memory_buffer.clear()

			# Learning rate annealing (implementation improvement from CleanRL [1])
			# [1] https://github.com/vwxyzjn/cleanrl	
			self._learningrate_annealing( update )			
			
			# The agent interacts with the enviornment for 'num-steps', notice that this number of steps
			# can be higher than a single episodes, in such case the environment is automatically reset by
			# as a feature of the vectorized wrapper
			for step in range(0, self.args.num_steps):

				# Compute the action and the probability of selecting the action with the neural network, these
				# values are part of the target for the objective function but do not require any information
				# about the gradient; we perform all these operations without tracking the gradient for a
				# faster computation 
				with torch.no_grad():
					actions, log_prob, _ = self.actor.get_action(state)
					value = self.critic.forward(state).view(-1).detach().numpy()
					cost_value = [cost_critic.forward(state).view(-1).detach().numpy() for cost_critic in self.cost_critics]				

				# Call the step function of the Gymnasium enviornment and extract the values for the (multiple)
				# cost functions; the flag is necessary because in the last step the vectorized setup
				# stores the information in a sub-dictionary before the automatic reset
				next_state, reward, terminated, truncated, infos = self.envs.step( actions.cpu().numpy() )
				if "cost" in infos: cost = infos["cost"]
				else: cost = numpy.array([final_info["cost"] for final_info in infos["final_info"]])

				# Preprocess the termination conditions, in a standard setup we can ignore the 
				# difference between 'terminated' and 'truncated'
				done = numpy.logical_or(terminated, truncated)

				# Minor fix for terminal states
				# for idx, d in enumerate(done): 
				# 	if d: next_state[idx] = infos["final_observation"][idx].copy()

				# Store data in the memory buffer for the network update
				memory_buffer.store_data( step, [state, actions, log_prob, reward, next_state, terminated, value] )
				memory_buffer.store_cost( step, [cost, cost_value] )

				# If the first environment terminates we store the performance in term of cost
				# and reward (n.b., recall that the first enviornment of the vectorized setup 
				# is our 'monitor' for the performance log)
				if done[0]: 
					cumulative_reward = infos['custom_info']['tot_reward'][0]
					ep_rewards[-1].append( cumulative_reward )
					cumulative_cost = infos['custom_info']["tot_costs"][0]
					ep_costs[-1].append( cumulative_cost )

				# Update the current states with the next states (standard
				# DRL loop for gym-like environments)
				state = torch.Tensor(next_state).to(self.device)

			# Log the performance after a complete update step, average the values in case of 
			# multiple episodes of the enviornment
			ep_rewards_queue.append( numpy.mean(ep_rewards[-1]) )
			ep_costs_queue.append( numpy.mean(ep_costs[-1], axis=0) )

			# Standard log of the results, for verbose=0 the results are only printed
			numpy_lambdas = [numpy.round(lag.detach().numpy(), 4) for lag in self.lagrangian_multipliers]
			print( f"Update number {update:3d}", end="" )
			print( f" reward: {ep_rewards_queue[-1]: 5.1f}", end="" )
			print( f" (average {numpy.mean(ep_rewards_queue): 5.1f})", end="" )
			print( f" cost: {[numpy.round(v, 1) for v in ep_costs_queue[-1]]}", end="" )
			print( f" (average {[numpy.round(v, 1) for v in numpy.mean(ep_costs_queue, axis=0)]})", end="" )
			print( f" lambda: {[numpy.round(v, 3) for v in numpy_lambdas]}")

			# Force output update (server only)
			sys.stdout.flush()

			# Performing the training steps after <args.num_steps> 
			self._train_networks( memory_buffer, ep_costs[-1] )

			# Log on WandB if required, i.e., only if verbose > 0
			if self.args.verbose > 0:
				record = { "reward": float(numpy.mean(ep_rewards[-1])) }
				for idx, cos in enumerate(ep_costs_queue[-1]): record[f"cost_{idx}"] = float(cos)
				for idx, lam in enumerate(numpy_lambdas): record[f"lambda_{idx}"] = float(lam)
				wandb.log(record)

		# Finalize wandb at the end of the training loop
		if self.args.verbose > 0: self._finalizeWandB()


	"""
		Mehtod that creates the enviornment, it is called multple time to generate the vector
		of environments for parallel execution. It also manage all the wrapper applied to the environment.
	"""
	def _make_env(self):
		def thunk():

			# Create the enviornment and add the basic custom wrappter 'MultiCostWrapper'; this is necessary
			# to fix the single-cost return and track the reward and cost values before normalization. For this 
			# last reason, it should always be applied before the normalization of costs and rewards
			env = gymnasium.make( self.args.env_id, render_mode=self.args.render_mode )
			env = MultiCostWrapper( env )
			
			# Application of some standard wrapper of gymnasium
			env = gymnasium.wrappers.FlattenObservation(env)
			env = gymnasium.wrappers.NormalizeObservation(env)
			env = gymnasium.wrappers.NormalizeReward(env, gamma=self.args.gamma)

			# This last wrapper is useful only for continuous action spaces
			if not isinstance(env.action_space, gymnasium.spaces.Discrete): env = gymnasium.wrappers.ClipAction(env)

			#
			return env

		return thunk 
	
	
	"""
		Mehtod that setups the information to be stored on WandB
	"""
	def _activateWandB( self ):

		# Compute the toal-steps parameters, this value should be 
		# de-normalized to abtain the real total, that is noe divided among
		# all the parallel enviornments
		self.args.total_steps = self.args.num_steps * self.args.num_envs 

		# Extract the information from the configuration file
		ymlfile = open("config/wandb_config.yaml", 'r')
		job_config = yaml.safe_load(ymlfile)
		yaml_entity = job_config["entity"]
		yaml_project = job_config["project"]
		yaml_tags = job_config["tags"]
		yaml_name = None if job_config["name"] == "None" else job_config["name"]
		yaml_mode = job_config["mode"]
		yaml_save_code = job_config["save_code"] == "True"

		# Initialize wandb with the correct paramters 
		wandb_path = wandb.init(
			name=yaml_name,
			tags=yaml_tags,
			entity=yaml_entity,
			project=yaml_project,
			mode=yaml_mode,
			save_code=yaml_save_code,
			config=self.args
		)

		# Store the record offile, they will be uploaded after the training if
		# mode=="offline", otherwise they will be uploaded at runtime
		self.wandb_path = os.path.split(wandb_path.dir)[0]


	"""
		Close the connection with WandB and send the colleceted data to the server
	"""
	def _finalizeWandB( self ):
		import subprocess
		wandb.finish()
		subprocess.run(['wandb', 'sync', self.wandb_path])


	"""
		Template for the '_learningrate_annealing' method tha must be implemented in the inherited class
	"""
	@abc.abstractmethod
	def _learningrate_annealing( self ): raise NotImplementedError


	"""
		Template for the '_train_networks' method tha must be implemented in the inherited class
	"""
	@abc.abstractmethod
	def _train_networks( self, memory_buffer ): raise NotImplementedError

	