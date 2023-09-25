import torch, numpy, gymnasium
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


"""
	Class that implements the memory buffer. This class implements some key methods for 
	the training loop. It manage the shapes of the given data (multiple enviornments, multiple steps,
	episodes, and multiple cost functions); it also perform some pre-processing steps, mainly the computation
	of GAE (a detailed explaination of GAE can be found in the original paper [1])

	[1] High-Dimensional Continuous Control Using Generalized Advantage Estimation. Schulman et al., 2018
"""
class MemoryBuffer( ):
	

	"""
		Constructor for the MemoryBuffer class, store all the paramters and initializes the actual 
		data structure for the buffer
	"""
	def __init__( self, num_steps, num_envs, envs, device, num_costs ):

		self.num_steps = num_steps
		self.num_envs = num_envs
		self.envs = envs
		self.device = device
		self.num_costs = num_costs

		self.buffer = self._new_buffer()

	
	"""
		Method that clear the buffer removing all the elements, notice that this operation must be done after
		each update for an 'on-policy' algorithm
	"""
	def clear( self ):
		self.buffer = self._new_buffer()


	"""
		Method that store the given data in the memory buffer, notice that it handle the shapes; in particular
		we use a matrix with 'step' rows (intially empty) and one column for each of parallel enviornment
		of the vectorized wrapper.
	"""
	def store_data( self, step, data ):
		self.buffer["state"][step] = data[0]
		self.buffer["action"][step] = data[1]
		self.buffer["log_prob"][step] = data[2]
		self.buffer["reward"][step] = torch.tensor(data[3]).to(self.device)
		self.buffer["next_state"][step] = torch.Tensor(data[4]).to(self.device)
		self.buffer["done"][step] = torch.Tensor(data[5]).to(self.device)
		self.buffer["value"][step] = torch.Tensor(data[6]).to(self.device).view(-1)	


	"""
		Method that store the given data related to cost, in addition to the standard 'store_data' method,
		here we have an additional dimention in the matrix to store the multiple cost functions supported
		by our approach
	"""
	def store_cost( self, step, data ):

		# Storing data in the memory buffer (step costs)
		costs = numpy.array(data[0].tolist())
		self.buffer["cost"][step] = torch.Tensor(costs).to(self.device)

		# Storing data in the memory buffer (step cost critics)
		cost_values = numpy.array(data[1]).T
		self.buffer["cost_value"][step] = torch.Tensor(cost_values).to(self.device)
			

	"""
		Computation of GAE for the advantage (reward), this implementation follows the suggested optimized
		algorithm from CleanRL [1], while for the theoretical dervation we refer to the original paper [2].

		[1] https://github.com/vwxyzjn/cleanrl
		[2] High-Dimensional Continuous Control Using Generalized Advantage Estimation. Schulman et al., 2018
	"""
	def add_GAE_reward( self, args, value_network ):
		
		with torch.no_grad():
			advantages = torch.zeros( (self.buffer["reward"].shape[0]+1, self.buffer["reward"].shape[1]) ).to(self.device)
			values_next_state = value_network.forward(self.buffer["next_state"]).reshape(self.num_steps, -1)
			
			for t in reversed(range(self.num_steps)):
				nextnonterminal = 1 - self.buffer["done"][t]
				delta = self.buffer["reward"][t] + args.gamma * values_next_state[t] * nextnonterminal - self.buffer["value"][t]
				advantages[t] = delta + (args.gamma * args.gae_lambda * advantages[t+1] * nextnonterminal)

			advantages = advantages[:-1, :]
		
		self.buffer["advantages"] = advantages

	
	"""
		Computation of GAE for the advantage (cost), this implementation follows the suggested optimized
		algorithm from CleanRL [1], while for the theoretical dervation we refer to the original paper [2].

		[1] https://github.com/vwxyzjn/cleanrl
		[2] High-Dimensional Continuous Control Using Generalized Advantage Estimation. Schulman et al., 2018
	"""
	def compute_GAE_cost_single( self, cost_buffer, cost_value_buffer, args, value_network ):

		with torch.no_grad():
			advantages = torch.zeros( (cost_buffer.shape[0]+1, cost_buffer.shape[1]) ).to(self.device)
			values_next_state = value_network.forward(self.buffer["next_state"]).reshape(self.num_steps, -1)
			
			for t in reversed(range(self.num_steps)):
				nextnonterminal = 1 - self.buffer["done"][t]
				delta = cost_buffer[t] + args.gamma * values_next_state[t] * nextnonterminal - cost_value_buffer[t]
				advantages[t] = delta + (args.gamma * args.gae_lambda * advantages[t+1] * nextnonterminal)

			advantages = advantages[:-1, :]

		return advantages
	
	
	"""
		This method simplify the computation for multiple cost advantage with GAE, basically it calls the standard 
		GAE estimation for the cost for a single function, iterating over all the functions in the buffer
	"""
	def add_GAE_cost( self, args, value_networks ):
		advantages = []
		for i in range( len(value_networks) ):
			advantage = self.compute_GAE_cost_single( self.buffer["cost"][:, :, i], self.buffer["cost_value"][:, :, i], args, value_networks[i]) 
			advantages.append( advantage )

		self.buffer["cost_advantages"] = advantages
	

	"""
		After the computation of the advantage with GAE it computes also the target for the computation of
		the loss function, it follows the optimization trick from [1]
		
		[1] https://github.com/vwxyzjn/cleanrl
	"""
	def add_target_return( self ):
		target_returns = self.buffer["advantages"] + self.buffer["value"]
		self.buffer["target_returns"] = target_returns
	

	"""
		After the computation of the cost advantage with GAE it computes also the target for the computation of
		the loss function, it follows the optimization trick from [1]. To support multiple cost functions
		we iterate over all the computed advantages
		
		[1] https://github.com/vwxyzjn/cleanrl
	"""
	def add_target_cost( self ):
		target_returns = []
		for i in range( len(self.buffer["cost_advantages"]) ):
			target_return = self.buffer["cost_advantages"][i] + self.buffer["cost_value"][:, :, i]
			target_returns.append( target_return )
		self.buffer["cost_target_returns"] = target_returns


	"""
		Method that flattens the memory buffer; this step is useful to consider all the experience from
		multiple environments as a unique buffer.
	"""
	def flatten_observation(self):

		# Pre process L-PPO data	
		buffer_cost, buffer_cost_advantages, buffer_cost_target_returns = [], [], []
		for i in range( len(self.buffer["cost_target_returns"]) ):
			buffer_cost.append( self.buffer["cost"][:, :, i].reshape(-1) )
			buffer_cost_advantages.append( self.buffer["cost_advantages"][i].reshape(-1) )
			buffer_cost_target_returns.append( self.buffer["cost_target_returns"][i].reshape(-1) )

		# Flatten PPO data
		self.buffer["state"] = self.buffer["state"].reshape( (-1,) + self.envs.single_observation_space.shape )
		self.buffer["action"] = self.buffer["action"].reshape( (-1,) + self.envs.single_action_space.shape )
		self.buffer["log_prob"] = self.buffer["log_prob"].reshape(-1)
		self.buffer["advantages"] = self.buffer["advantages"].reshape(-1)
		self.buffer["target_returns"] = self.buffer["target_returns"].reshape(-1)
		
		# Flatten L-PPO data
		self.buffer["cost"] = buffer_cost
		self.buffer["cost_advantages"] = buffer_cost_advantages
		self.buffer["cost_target_returns"] = buffer_cost_target_returns


	"""
		Method that just returns the relevant information related to the reward before the optimization step
	"""
	def get_update_data( self ):
		return self.buffer["state"], self.buffer["action"], self.buffer["log_prob"], \
			self.buffer["advantages"], self.buffer["target_returns"]
	
	
	"""
		Method that just returns the relevant information related to the cost before the optimization step
	"""
	def get_cost_update_data( self ):
		return self.buffer["cost_advantages"], self.buffer["cost_target_returns"]


	"""
		Method that generates the actual data structure to store the data with multiple shapes. We use a matrix 
		with 'step' rows (intially empty) and one column for each of parallel enviornment of the vectorized 
		wrapper; for the costwe have an additional dimention in the matrix to store the multiple cost functions 
		supported by our approach
	"""
	def _new_buffer( self ):

		# Standard 2-dimensions matrix (i.e., 'steps'*'num-environments')
		buffer = { 
			"state": torch.zeros( (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape ).to(self.device), 
			"action": torch.zeros( (self.num_steps, self.num_envs) + self.envs.single_action_space.shape ).to(self.device), 
			"log_prob": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device), 
			"reward": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device), 
			"next_state": torch.zeros( (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape ).to(self.device),
			"done": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device),
			"value": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device) 
		}

		# Extended 3-dimensions matrix for the cost (i.e., 'steps'*'num-environments'*'num-cost-functions')
		buffer["cost"] = torch.zeros( (self.num_steps, self.num_envs, self.num_costs) ).to(self.device)
		buffer["cost_value"] = torch.zeros( (self.num_steps, self.num_envs, self.num_costs) ).to(self.device)

		return buffer


"""
	Class that implements the policy neural network; the genral structure is 3-layer DNN of 64 neurons with
	'tanh' as activation function. The input and output layer sizes are computed automatically extracting the
	data from the given environment.
"""
class PolicyNetwork( torch.nn.Module ):


	"""
		Constructor for the PolicyNetwork class, inherited from torch standard DNN class
	"""
	def __init__( self, envs ):
		super().__init__()
		
		# Creation of the network topology
		self.l1 = torch.nn.Linear(numpy.array(envs.single_observation_space.shape).prod(), 64)
		self.l2 = torch.nn.Tanh()
		self.l3 = self._layer_init( torch.nn.Linear(64, 64) )
		self.l4 = torch.nn.Tanh()
		self.l5 = self._layer_init( torch.nn.Linear(64, numpy.prod(envs.single_action_space.shape)), std=0.01 )
		self.lO = torch.nn.Tanh()

		# Additional Variable for exploration (variance), remember that the policy network for continuous action
		# spaces implements a Gaussian distribution. We build a structure with multiple means (one for each degree 
		# of freedom, i.e., the action space) and one single output for the variance.
		# NB: the std-deviation is returned in the logarithmic form to simplify the computation
		log_std = numpy.log(0.9) * numpy.ones(numpy.prod(envs.single_action_space.shape), dtype=numpy.float32)
		self.actor_logstd = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)


	"""
		Method that perform the forward computation
	"""	
	def forward( self, x ):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		output = self.l5(x)

		return self.lO(output)
	

	"""
		Method that exploits the forward computation to compute the action to perform, the 
		action is selected by sampling from the Gaussian distribution generated exploiting
		the 'mean' returned for each action and the std-deviation that is shared for all the actions.

		NB: after the training process the std-deviation should reduce over time (if everything works properly!)
		and so in the last stages of the training the action should be just the average
	"""	
	def get_action( self, x, action=None ):

		# Compute the means, one for each action to return
		normal_means = self.forward( x )

		# Get the current standard deviation, the expansion is for simplicity, basically
		# we replicate the same std-deviation for each mean to obtain 'N' normal distributions
		normal_logstd = self.actor_logstd.expand_as(normal_means)

		# Use the exponential function to recompute the real std-deviation (recall that 
		# we returned it in logarithmic form); we then generate a normal distribution with torch
		normal_std = torch.exp(normal_logstd)		
		probs = Normal(normal_means, normal_std)
		
		# If not given, we sample the probability from the distribution, this check is useful
		# for the optimization step! In some cases we want to just compute the probability of 
		# a given action, without sampling it
		if action is None: action = probs.sample()

		# In case of multiple environments, we consider the joint probability of selecting the given actions
		# which basically is the multiplication among the probability of selecting the actions. Given that we 
		# are working with the log-prob, the joint probability is the sum of all the probabilities.
		# TL; DR: the 'sum(1)' fix the multiple-environment of the vectorized wrapper
		return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


	"""
		The orthogonal initialization is an optimization trick from [1]
		[1] https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
	"""
	def _layer_init( self, layer, std=numpy.sqrt(2), bias_const=0.0 ):
		torch.nn.init.orthogonal_( layer.weight, std )
		torch.nn.init.constant_( layer.bias, bias_const )
		return layer
	

"""
	Class that implements the value function neural network; the genral structure is 2-layer DNN of 64 neurons with
	'tanh' as activation function. The input layer size is computed automatically extracting the information
	from the given environment.
"""
class ValueNetwork( torch.nn.Module ):


	"""
		Constructor for the PolicyNetwork class, inherited from torch standard DNN class
	"""
	def __init__( self, envs ):
		super().__init__()
		
		# Creation of the network topology, notice that the output layer is always of
		# 1 element because the value is a single value given a specific state
		self.l1 = torch.nn.Linear(numpy.array(envs.single_observation_space.shape).prod(), 64)
		self.l2 = torch.nn.Tanh()
		self.l3 = self._layer_init( torch.nn.Linear(64, 64) )
		self.l4 = torch.nn.Tanh()
		self.l5 = self._layer_init( torch.nn.Linear(64, 1), std=1.0 )		


	"""
		Method that perform the forward computation
	"""	
	def forward( self, x ):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		output = self.l5(x)

		return output
	

	"""
		The orthogonal initialization is an optimization trick from [1]
		[1] https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
	"""
	def _layer_init( self, layer, std=numpy.sqrt(2), bias_const=0.0 ):
		torch.nn.init.orthogonal_( layer.weight, std )
		torch.nn.init.constant_( layer.bias, bias_const )
		return layer


"""
	Class that implements the policy neural network for the special case of DISCRETE ACTION SPACE; 
	the genral structure is 3-layer DNN of 64 neurons with 'tanh' as activation function. The input 
	and output layer sizes are computed automatically extracting the data from the given environment.
"""
class DiscretePolicyNetwork( torch.nn.Module ):


	"""
		Constructor for the PolicyNetwork class, inherited from torch standard DNN class
	"""
	def __init__( self, envs ):
		super().__init__()
		
		# Creation of the network topology, notice that here we don't need a std-deviation
		# beacuse the output already represent a probability distribution over all the possible
		# discrete actions and we can directly sample from it.
		# NB: In the last layer we don't have a softmax activation, this is an implementation 
		# approach, we generate the distribution from the real values after the propagation
		self.l1 = torch.nn.Linear(numpy.array(envs.single_observation_space.shape).prod(), 64)
		self.l2 = torch.nn.Tanh()
		self.l3 = self._layer_init( torch.nn.Linear(64, 64) )
		self.l4 = torch.nn.Tanh()
		self.l5 = self._layer_init( torch.nn.Linear(64, numpy.prod(envs.single_action_space.n)), std=0.01 )


	"""
		Method that perform the forward computation
	"""	
	def forward( self, x ):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		output = self.l5(x)
		
		return output
	

	"""
		Method that exploits the forward computation to compute the action to perform, the 
		action is selected by sampling from distribution represented in the output layer over
		all the possibile discrete actions
	"""	
	def get_action( self, x, action=None ):

		# Generation of the categorical distribution over
		# all the possible discrete actions, the 'categorical' torch
		# function allows us to avoid the 'softmax' beacuse it automatically generate
		# a distribution from real values
		logits = self.forward( x )
		probs = Categorical(logits=logits)

		# If not given, we sample the probability from the distribution, this check is useful
		# for the optimization step! In some cases we want to just compute the probability of 
		# a given action, without sampling it
		if action is None: action = probs.sample()

		# Return the action, probabilities and entropys
		return action, probs.log_prob(action), probs.entropy()


	"""
		The orthogonal initialization is an optimization trick from [1]
		[1] https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
	"""
	def _layer_init( self, layer, std=numpy.sqrt(2), bias_const=0.0 ):
		torch.nn.init.orthogonal_( layer.weight, std )
		torch.nn.init.constant_( layer.bias, bias_const )
		return layer

