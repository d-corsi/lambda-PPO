import gymnasium, safety_gymnasium
import torch, numpy, random
import collections, abc, wandb, os, sys
from scripts.utils import PolicyNetwork, DiscretePolicyNetwork, ValueNetwork, MemoryBuffer


class NormalizeCost( gymnasium.wrappers.NormalizeReward ):
    
	def step(self, action):
		obs, rews, terminateds, truncateds, infos = self.env.step(action)
		infos["original_cost"] = infos["cost"]
		cost = [infos["cost"]]		
		self.returns = self.returns * self.gamma * (1 - terminateds) + cost
		cost = self.normalize(cost)
		infos["cost"] = cost[0]
		return obs, rews, terminateds, truncateds, infos
	

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
		# self.envs = safety_gymnasium.vector.make( self.args.env_id, render_mode=None, num_envs=self.args.num_envs )
		self.envs = gymnasium.vector.AsyncVectorEnv( [self._make_env() for _ in range(self.args.num_envs) ] )

		# Sanity check on the episodes for the circle environment
		if self.args.env_id[-10:-4] == "Circle" and self.args.num_steps < 500: raise ValueError( "Increase the total number of steps" )
		elif self.args.num_steps < 1000: raise ValueError( "Increase the total number of steps" )

		# Initialize the neural networks
		if isinstance(self.envs.single_action_space, gymnasium.spaces.Discrete): self.actor = DiscretePolicyNetwork( self.envs ).to(self.device)
		else: self.actor = PolicyNetwork( self.envs ).to(self.device)
		self.critic = ValueNetwork( self.envs ).to(self.device)
		self.cost_critic = ValueNetwork( self.envs ).to(self.device)

		# Initialize the optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.learning_rate, eps=1e-5)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.learning_rate, eps=1e-5)
		self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.args.learning_rate, eps=1e-5)

		# Initialize the optimizers for the lagrangian multiplier		
		self.lagrangian_multiplier = torch.nn.Parameter( torch.as_tensor(self.args.lambda_init), requires_grad=True )
		self.lambda_optimizer = torch.optim.Adam( [self.lagrangian_multiplier], lr=self.args.lambda_learning_rate )
		

	def main_loop( self ):

		# Activate wandb to save the logging data (if required)
		if self.args.verbose > 0: self._activateWandB()

		# Create the memory buffer object for the training loop
		memory_buffer = MemoryBuffer(  self.args.num_steps, self.args.num_envs, self.envs, self.device )

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
			ep_cost = 0

			# Memory buffer init; policy gradient methods require the reset after each policy update
			memory_buffer.clear()

			# Learning rate annealing
			self._learningrate_annealing( update )			
			
			# Starting the episode for <args.num_steps> steps
			for step in range(0, self.args.num_steps):

				with torch.no_grad():
					actions, log_prob, _ = self.actor.get_action(state)
					value = self.critic.forward(state).view(-1)
					cost_value = self.cost_critic.forward(state).view(-1) 

				# next_state, reward, cost, terminated, truncated, infos = self.envs.step( actions.cpu().numpy() )
				next_state, reward, terminated, truncated, infos = self.envs.step( actions.cpu().numpy() )

				# Preprocess some data
				done = numpy.logical_or(terminated, truncated)
				if done[0]: cost = [infos["final_info"][idx]["cost"] for idx in range(self.args.num_envs)]
				else: cost = infos["cost"]

				# Store data in the memory buffer
				memory_buffer.store_data( step, [state, actions, log_prob, reward, next_state, done, value] )
				memory_buffer.store_cost( step, [cost, cost_value] )

				# In this version we skip the last cost, it doesn't matter since it's only for debugging
				# purpose but can miss a +/- 1. Notice that the value returned by the custom wrapper is 
				# still correct.
				# if not done[0]: ep_cost += infos["original_cost"][0]
				ep_cost += cost

				if done[0]: 
					ep_rewards[-1].append( infos['final_info'][0]['episode']['r'][0] )
					ep_costs[-1].append( ep_cost )
					ep_cost = 0				

				state = torch.Tensor(next_state).to(self.device)

			# End Episode Logging
			ep_rewards_queue.append( numpy.mean(ep_rewards[-1]) )
			ep_costs_queue.append( numpy.mean(ep_costs[-1]) )

			print( f"Update number {update:3d}", end="" )
			print( f" reward: {ep_rewards_queue[-1]: 5.1f}", end="" )
			print( f" (average {numpy.mean(ep_rewards_queue): 5.1f})", end="" )
			print( f" cost: {int(ep_costs_queue[-1]):3d}", end="" )
			print( f" (average {numpy.mean(ep_costs_queue): 3.1f})", end="" )
			print( f" lambda: {self.lagrangian_multiplier.detach().numpy():5.4f}")

			# Force output update (server only)
			sys.stdout.flush()

			# Performing the training steps after <args.num_steps> 
			self._train_networks( memory_buffer, ep_rewards[-1], ep_costs[-1] )

			# Log on WandB if required
			if self.args.verbose > 0:
				record = {
					"reward": float(numpy.mean(ep_rewards[-1])),
					"cost": float(numpy.mean(ep_costs[-1])),
					"lambda": float(self.lagrangian_multiplier.detach().numpy())
				}
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

			gymnasium_name = self.args.env_id[:-3] + "Gymnasium" + self.args.env_id[-3:]

			env = gymnasium.make( gymnasium_name, render_mode="rgb_array" )

			# env = gymnasium.wrappers.RecordVideo(env, video_folder="videos")
			
			env = gymnasium.wrappers.FlattenObservation(env)
			env = gymnasium.wrappers.RecordEpisodeStatistics(env)
			env = gymnasium.wrappers.FlattenObservation(env)
			env = gymnasium.wrappers.NormalizeObservation(env)
			env = gymnasium.wrappers.NormalizeReward(env, gamma=self.args.gamma)
			# env = NormalizeCost(env, gamma=self.args.gamma)

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