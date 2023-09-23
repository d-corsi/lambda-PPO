from scripts.utils import PolicyNetwork, DiscretePolicyNetwork, ValueNetwork, MemoryBuffer
from scripts.wrappers import MultiCostWrapper
import gymnasium, safety_gymnasium, torch, numpy, random
import collections, abc, wandb, os, sys
	

class ReinforcementLearning( abc.ABC ):
    
	def __init__(self, args ):

		self.args = args
		self.current_update = 0
		
		# Seeding
		random.seed( self.args.seed )
		numpy.random.seed( self.args.seed )
		torch.manual_seed( self.args.seed )

		#
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		#
		self.envs = gymnasium.vector.AsyncVectorEnv( [self._make_env() for _ in range(self.args.num_envs) ] )

		# Initialize the neural networks for actor and critic (reward-wise)
		if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete): self.actor = DiscretePolicyNetwork( self.envs ).to(self.device)
		else: self.actor = PolicyNetwork( self.envs ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.learning_rate, eps=1e-5)
		self.critic = ValueNetwork( self.envs ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.learning_rate, eps=1e-5)
		
		# Initialize the neural networks for the critics (cost-wise)
		self.cost_critics = [ValueNetwork( self.envs ).to(self.device) for _ in range(self.args.num_costs)]
		self.cost_critic_optimizers = [torch.optim.Adam(self.cost_critics[i].parameters(), lr=self.args.learning_rate, eps=1e-5) for i in range(self.args.num_costs)]

		# Initialize the optimizers for the lagrangian multiplier		
		self.lagrangian_multipliers = [torch.nn.Parameter( torch.as_tensor(self.args.lambda_init), requires_grad=True ) for _ in range(self.args.num_costs)]
		self.lambda_optimizers = [torch.optim.Adam( [self.lagrangian_multipliers[i]], lr=self.args.lambda_learning_rate ) for i in range(self.args.num_costs)]
		

	def main_loop( self ):

		# Activate wandb to save the logging data (if required)
		if self.args.verbose > 0: self._activateWandB()

		# Create the memory buffer object for the training loop
		memory_buffer = MemoryBuffer(  self.args.num_steps, self.args.num_envs, self.envs, self.device, self.args.num_costs )

		# Initialize logging data
		ep_costs = []
		ep_rewards = []
		ep_costs_queue = collections.deque(maxlen=100)
		ep_rewards_queue = collections.deque(maxlen=100)


		for update in range(1, (self.args.updates+1)):

			# Reset the environment before starting the process
			state, _ = self.envs.reset( seed=self.args.seed )
			state = torch.Tensor(state).to(self.device)

			self.current_update = update

			# Counters for the single agent
			ep_costs.append([])
			ep_rewards.append([])

			# Memory buffer init; policy gradient methods require the reset after each policy update
			memory_buffer.clear()

			# Learning rate annealing
			self._learningrate_annealing( update )			
			
			# Starting the episode for <args.num_steps> steps
			for step in range(0, self.args.num_steps):

				with torch.no_grad():
					actions, log_prob, _ = self.actor.get_action(state)
					value = self.critic.forward(state).view(-1).detach().numpy()
					cost_value = [cost_critic.forward(state).view(-1).detach().numpy() for cost_critic in self.cost_critics]				

				#
				next_state, reward, terminated, truncated, infos = self.envs.step( actions.cpu().numpy() )
				if "cost" in infos: cost = infos["cost"]
				else: cost = numpy.array([final_info["cost"] for final_info in infos["final_info"]])

				# Preprocess some data
				done = numpy.logical_or(terminated, truncated)

				# Store data in the memory buffer
				memory_buffer.store_data( step, [state, actions, log_prob, reward, next_state, done, value] )
				memory_buffer.store_cost( step, [cost, cost_value] )

				if done[0]: 
					cumulative_reward = infos['final_info'][0]['custom_info']['tot_reward']
					ep_rewards[-1].append( cumulative_reward )
					cumulative_cost = infos['final_info'][0]['custom_info']["tot_costs"]
					ep_costs[-1].append( cumulative_cost )

				state = torch.Tensor(next_state).to(self.device)

			# End Episode Logging
			ep_rewards_queue.append( numpy.mean(ep_rewards[-1]) )
			ep_costs_queue.append( numpy.mean(ep_costs[-1], axis=0) )

			# Data logs
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
			self._train_networks( memory_buffer, ep_rewards[-1], ep_costs[-1] )

			# Log on WandB if required
			if self.args.verbose > 0:
				record = { "reward": float(numpy.mean(ep_rewards[-1])) }
				for idx, cos in enumerate(ep_costs_queue[-1]): record[f"cost_{idx}"] = float(cos)
				for idx, lam in enumerate(numpy_lambdas): record[f"lambda_{idx}"] = float(lam)
				wandb.log(record)

		# Finalize wandb at the end of the training loop
		if self.args.verbose > 0: self._finalizeWandB()


	def _activateWandB( self ):
		self.args.total_steps = self.args.num_steps * self.args.num_envs 

		wandb_path = wandb.init(
			name=None,
			tags=["params"],
			entity="dcorsi",
			project="lambda-PPO",
			mode="offline",
			save_code=False,
			config=self.args
		)
		self.wandb_path = os.path.split(wandb_path.dir)[0]


	def _make_env(self):
		def thunk():

			env = gymnasium.make( self.args.env_id, render_mode=self.args.render_mode )
			env = MultiCostWrapper( env )
			
			#
			env = gymnasium.wrappers.FlattenObservation(env)
			env = gymnasium.wrappers.NormalizeObservation(env)
			env = gymnasium.wrappers.NormalizeReward(env, gamma=self.args.gamma)

			#
			if not isinstance(env.action_space, gymnasium.spaces.Discrete): env = gymnasium.wrappers.ClipAction(env)

			return env

		return thunk 


	def _finalizeWandB( self ):
		import subprocess
		wandb.finish()
		subprocess.run(['wandb', 'sync', self.wandb_path])


	@abc.abstractmethod
	def _learningrate_annealing( self ): raise NotImplementedError


	@abc.abstractmethod
	def _train_networks( self, memory_buffer ): raise NotImplementedError