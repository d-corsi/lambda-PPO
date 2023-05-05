import torch, numpy
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class MemoryBuffer( ):
	
	def __init__( self, num_steps, num_envs, envs, device ):

		self.num_steps = num_steps
		self.num_envs = num_envs
		self.envs = envs
		self.device = device

		self.buffer = self._new_buffer()

	
	def clear( self ):
		self.buffer = self._new_buffer()


	def store_data( self, step, data ):
		# Storing data in the memory buffer (PPO)
		self.buffer["state"][step] = data[0]
		self.buffer["action"][step] = data[1]
		self.buffer["log_prob"][step] = data[2]
		self.buffer["reward"][step] = torch.tensor(data[3]).to(self.device)
		self.buffer["next_state"][step] = torch.Tensor(data[4]).to(self.device)
		self.buffer["done"][step] = torch.Tensor(data[5]).to(self.device)
		self.buffer["value"][step] = torch.Tensor(data[6]).to(self.device).view(-1)	


	def store_cost( self, step, data ):
		# Storing data in the memory buffer (L-PPO)
		self.buffer["cost"][step] = torch.Tensor(data[0]).to(self.device)
		self.buffer["cost_value"][step] = torch.Tensor(data[1]).to(self.device)
			

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


	def add_GAE_cost( self, args, value_network ):
		
		with torch.no_grad():
			advantages = torch.zeros( (self.buffer["cost"].shape[0]+1, self.buffer["cost"].shape[1]) ).to(self.device)
			values_next_state = value_network.forward(self.buffer["next_state"]).reshape(self.num_steps, -1)
			
			for t in reversed(range(self.num_steps)):
				nextnonterminal = 1 - self.buffer["done"][t]
				delta = self.buffer["cost"][t] + args.gamma * values_next_state[t] * nextnonterminal - self.buffer["cost_value"][t]
				advantages[t] = delta + (args.gamma * args.gae_lambda * advantages[t+1] * nextnonterminal)

			advantages = advantages[:-1, :]
		
		self.buffer["cost_advantages"] = advantages


	def add_target_return( self ):
		target_returns = self.buffer["advantages"] + self.buffer["value"]
		self.buffer["target_returns"] = target_returns
	
	
	def add_target_cost( self ):
		target_returns = self.buffer["cost_advantages"] + self.buffer["cost_value"]
		self.buffer["cost_target_returns"] = target_returns


	def flatten_observation(self):
		# Flatten PPO data
		self.buffer["state"] = self.buffer["state"].reshape( (-1,) + self.envs.single_observation_space.shape )
		self.buffer["action"] = self.buffer["action"].reshape( (-1,) + self.envs.single_action_space.shape )
		self.buffer["log_prob"] = self.buffer["log_prob"].reshape(-1)
		self.buffer["advantages"] = self.buffer["advantages"].reshape(-1)
		self.buffer["target_returns"] = self.buffer["target_returns"].reshape(-1)
		# Flatten L-PPO data
		self.buffer["cost"] = self.buffer["cost"].reshape(-1)
		self.buffer["cost_advantages"] = self.buffer["cost_advantages"].reshape(-1)
		self.buffer["cost_target_returns"] = self.buffer["cost_target_returns"].reshape(-1)


	def get_update_data( self ):
		return self.buffer["state"], self.buffer["action"], self.buffer["log_prob"], \
			self.buffer["advantages"], self.buffer["target_returns"]
	
	
	def get_cost_update_data( self ):
		return self.buffer["cost_advantages"], self.buffer["cost_target_returns"]


	def _new_buffer( self ):

		buffer = { 
			"state": torch.zeros( (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape ).to(self.device), 
			"action": torch.zeros( (self.num_steps, self.num_envs) + self.envs.single_action_space.shape ).to(self.device), 
			"log_prob": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device), 
			"reward": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device), 
			"next_state": torch.zeros( (self.num_steps, self.num_envs) + self.envs.single_observation_space.shape ).to(self.device),
			"done": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device),
			"value": torch.zeros( (self.num_steps, self.num_envs) ).to(self.device) 
		}

		# Add a key for each cost function
		buffer["cost"] = torch.zeros( (self.num_steps, self.num_envs) ).to(self.device)
		buffer["cost_value"] = torch.zeros( (self.num_steps, self.num_envs) ).to(self.device)

		return buffer


class PolicyNetwork( torch.nn.Module ):

	def __init__( self, envs ):
		super().__init__()
		
		self.l1 = torch.nn.Linear(numpy.array(envs.single_observation_space.shape).prod(), 64)
		self.l2 = torch.nn.Tanh()
		self.l3 = self._layer_init( torch.nn.Linear(64, 64) )
		self.l4 = torch.nn.Tanh()
		self.l5 = self._layer_init( torch.nn.Linear(64, numpy.prod(envs.single_action_space.shape)), std=0.01 )
		self.lO = torch.nn.Tanh()

		# Additional Variable for exploration (variance)
		# self.actor_logstd = torch.nn.Parameter(torch.zeros(-1.2, numpy.prod(envs.single_action_space.shape)))
		log_std = numpy.log(0.9) * numpy.ones(numpy.prod(envs.single_action_space.shape), dtype=numpy.float32)
		self.actor_logstd = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)


	def forward( self, x ):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		output = self.l5(x)

		output = self.lO(output)
		
		return output
	

	def get_action( self, x, action=None ):
		normal_means = self.forward( x )
		normal_logstd = self.actor_logstd.expand_as(normal_means)
		normal_std = torch.exp(normal_logstd)		
		probs = Normal(normal_means, normal_std)
		
		if action is None: 
			action = probs.sample()

		return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


	def _layer_init( self, layer, std=numpy.sqrt(2), bias_const=0.0 ):
		torch.nn.init.orthogonal_( layer.weight, std )
		torch.nn.init.constant_( layer.bias, bias_const )
		return layer
	

class ValueNetwork( torch.nn.Module ):

	def __init__( self, envs ):
		super().__init__()
		
		self.l1 = torch.nn.Linear(numpy.array(envs.single_observation_space.shape).prod(), 64)
		self.l2 = torch.nn.Tanh()
		self.l3 = self._layer_init( torch.nn.Linear(64, 64) )
		self.l4 = torch.nn.Tanh()
		self.l5 = self._layer_init( torch.nn.Linear(64, 1), std=1.0 )		

	def forward( self, x ):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		output = self.l5(x)

		return output
	

	def _layer_init( self, layer, std=numpy.sqrt(2), bias_const=0.0 ):
		torch.nn.init.orthogonal_( layer.weight, std )
		torch.nn.init.constant_( layer.bias, bias_const )
		return layer


class DiscretePolicyNetwork( torch.nn.Module ):

	def __init__( self, envs ):
		super().__init__()
		
		self.l1 = torch.nn.Linear(numpy.array(envs.single_observation_space.shape).prod(), 64)
		self.l2 = torch.nn.Tanh()
		self.l3 = self._layer_init( torch.nn.Linear(64, 64) )
		self.l4 = torch.nn.Tanh()
		self.l5 = self._layer_init( torch.nn.Linear(64, numpy.prod(envs.single_action_space.n)), std=0.01 )


	def forward( self, x ):
		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		output = self.l5(x)
		
		return output
	

	def get_action( self, x, action=None ):
		logits = self.forward( x )
		probs = Categorical(logits=logits)

		if action is None: 
			action = probs.sample()

		return action, probs.log_prob(action), probs.entropy()


	def _layer_init( self, layer, std=numpy.sqrt(2), bias_const=0.0 ):
		torch.nn.init.orthogonal_( layer.weight, std )
		torch.nn.init.constant_( layer.bias, bias_const )
		return layer