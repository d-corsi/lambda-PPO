from scripts.DRL import ReinforcementLearning
import numpy, torch


"""
	Class that implements the λ-PPO update rules. It optimizes a single reward function but 
	supports multiple cost functions. The lagrangian multipliers are optimized with gradient-descent
	as well as the policy and value functons. The normalization strategy for the multipliers
	follows an hybrid strategy between [1] and [2]: the sum of all the multipliers can not be
	greater than '1-min_lambda_reward', in such case the multipliers are normalized with a softmax function.
	
	The final update rules will be:
		Loss = λ_r * reward - sum_i( λ_i * cost_i ) 			> where λ_r = 1-sum_(λ_i)

	[1] Direct behavior specification via constrained reinforcement learning. Roy et al. 2021
	[2] Constrained reinforcement learning for robotics via scenario-based programming. Corsi et al. 2022
"""
class LambdaPPO( ReinforcementLearning ):
    

	"""
		Constructor for the LambdaPPO class, inherited from the parent class.
	"""
	def __init__( self, args ):
		super().__init__( args )

	
	"""
		Method that implements the '_learningrate_annealing' techinques to reduce thi paramter over time,
		this is an implementation trick from CleanRL [1]
		
		[1] https://github.com/vwxyzjn/cleanrl
	"""
	def _learningrate_annealing( self, update ):

		# Scaling the learing rate to obtain a linear decay over the total 
		# number of update steps
		frac = 1.0 - (update - 1.0) / self.args.updates
		lrnow = frac * self.args.learning_rate

		# Assign the new value to all the optimizer (actor, reward critic and cost critics); the
		# learning rate for the lagrangian multipliers is not changed by this function
		self.actor_optimizer.param_groups[0]["lr"] = lrnow
		self.critic_optimizer.param_groups[0]["lr"] = lrnow 
		for cost_critic_optimizer in self.cost_critic_optimizers: cost_critic_optimizer.param_groups[0]["lr"] = lrnow


	"""
		Main method that implements the update rules for all the components of the λ-PPO algorithm: actor network,
		critic network, cost-critic networks, and lagrangian multipliers (the reward multiplier is updated automatically).
		
	"""
	def _train_networks( self, memory_buffer, avg_cost=None ):

		# Pre-preprocess the data (with GAE) and get the data from the memory buffer;
		# the 'flattening' step is necessary for the parallel execution of multiple environments,
		# this step consider all these experience as a unique buffer
		memory_buffer.add_GAE_reward( self.args, self.critic )
		memory_buffer.add_GAE_cost( self.args, self.cost_critics )
		memory_buffer.add_target_return()
		memory_buffer.add_target_cost()
		memory_buffer.flatten_observation()

		# Extract the relevant infos from the buffer
		states, actions, log_probs, advantages, target_returns = memory_buffer.get_update_data()
		cost_advantages, cost_target_returns = memory_buffer.get_cost_update_data()

		# Get 'all' the indices of all the element of the buffer, this will be useful to split
		# the elements of the buffer into minibatch 
		buffer_indices = numpy.arange( self.args.num_steps * self.args.num_envs )

		# STEP 1: update of the lagrangian multipliers, this step is based on the episode costs
		# and update all the multipliers together, in this step there is also performed the
		# normalization with the 'softmax' technique
		if self.current_update > self.args.start_train_lambda:

			# All the episode costs are subdivided in minibatch and averaged betweem multiple experiences,
			# we then call the update function based on the gradient
			cost_batches = numpy.array_split(avg_cost, min(self.args.lambda_batch_number, len(avg_cost)))
			cost_batch = [numpy.mean(b, axis=0) for b in cost_batches]
			for cost in cost_batch: self.update_lambda(cost)

			# Force the lower bound to be greater than zero; this is a theoretical requirement
			# from the lagrangian dual relaxation
			for i in range(self.args.num_costs): self.lagrangian_multipliers[i].data.clamp_( min=0 )

			# If the sum of the lagrangian multipliers exceed the limit imposed by the paramter 'min_lambda_reward'
			# (or just is greater than 1), here we perform the normalization step exploiting the 'softmax' technique [1]
			# [1] Direct behavior specification via constrained reinforcement learning. Roy et al. 2021
			lambdas = numpy.array([lag.detach().numpy() for lag in self.lagrangian_multipliers])
			if sum(lambdas) > 1 - self.args.min_lambda_reward: 
				lambdas = self.softmax( lambdas, temperature=0.3, total=(1-self.args.min_lambda_reward) )

			# To force the normalized value without affecting the gradient, we exploit the 'clamp' function from torch
			for i in range(self.args.num_costs): self.lagrangian_multipliers[i].data.clamp_( min=lambdas[i], max=lambdas[i])

		# STEP 2: update of of the value functions, this step update all the critics (reward and costs)
		for _ in range( self.args.value_epochs ):

			# For each update step we perform a shuffle of the indices before splitting the buffer into multiple
			# batches; this allows to reduce the correlation. Notice that we lose the temporal correlation but this is 
			# not a problem because the value function is update 'off-policy'.
			numpy.random.shuffle( buffer_indices )
			batches = numpy.array_split( buffer_indices, self.args.value_batch_number )

			# We iterate over all the minibatches to update the networks
			for minibatch in batches:

				# Update the advantage value function, first we compute the loss and then we perform 
				# the gradient descent step
				value_loss = self.get_value_loss( self.critic, states[minibatch], target_returns[minibatch] )
				self.optimization_step( self.critic, self.critic_optimizer, value_loss )

				# Update the multiple cost value functions (iterate over all of them), first we compute the loss 
				# and then we perform the gradient descent step
				for i in range(self.args.num_costs):
					cost_value_loss = self.get_value_loss( self.cost_critics[i], states[minibatch], cost_target_returns[i][minibatch] )
					self.optimization_step( self.cost_critics[i], self.cost_critic_optimizers[i], cost_value_loss )


		# STEP 3: update of of the poloicy functions, this step update the actor network
		for vep in range( self.args.epochs ):

			# For each update step we perform a shuffle of the indices before splitting the buffer into multiple
			# batches; this allows to reduce the correlation. Notice that we lose the temporal correlation but this is 
			# not a problem given the actor critic architecture. This is ok only if we clear the buffer after each update 
			# step, otherwise we violate the requirement for the 'on-policy' update of the actor
			numpy.random.shuffle( buffer_indices )
			batches = numpy.array_split( buffer_indices, self.args.batch_number )

			# We iterate over all the minibatches to update the actor network
			for minibatch in batches:

				# Comptutation of the loss functions and kl-dicergence (the latter is for the 'early exit', an 
				# implementation trick from [1].
				# [1] https://spinningup.openai.com/en/latest/
				policy_loss, approx_kl = self.get_policy_loss_and_kl( states[minibatch], actions[minibatch], \
							 log_probs[minibatch], advantages[minibatch], [ca[minibatch] for ca in cost_advantages] )
				
				# Perform the actual optimization step with gradient ascent
				self.optimization_step( self.actor, self.actor_optimizer, policy_loss )
				
			# Check the last kl-divergence (supposed to be the highest), if higher than the limit
			# we perform the early stop
			if (approx_kl.mean().cpu().detach().numpy()) > self.args.target_kl: 
				# print( f"\tDEBUG early stop after iteration {vep}/{self.args.epochs} for kl-divergence" )
				break
				
		# print( "======" )
		# _, _, entropy = self.actor.get_action( states, action=actions  )
		# print( f"\tDEBUG: entropy {entropy.mean().detach().numpy():5.3f}" )
		# print( f"\tDEBUG: gaussian std {numpy.exp(self.actor.actor_logstd.detach().numpy())}" )
		# print( f"\tDEBUG: value loss {value_loss:5.3f}" )
		# print( f"\tDEBUG: cost value loss {cost_value_loss:5.3f}" )
		# print( f"\tDEBUG: lambda loss {lambda_loss.detach().numpy()[0]:5.3f}" )
		# print( f"\tDEBUG: policy loss {policy_loss.detach().numpy():5.3f}" )
		# print( f"\tDEBUG: average actions {numpy.mean(actions.detach().numpy(), axis=0)}" )
		# print( f"\tDEBUG: min actions {numpy.min(actions.detach().numpy(), axis=0)}" )
		# print( f"\tDEBUG: max actions {numpy.max(actions.detach().numpy(), axis=0)}" )
		# print( "reward-advantage", numpy.mean(advantages.detach().numpy()) )
		# print( "cost-advantage", numpy.mean(cost_advantages.detach().numpy()) )
		# print( "======" )


	"""
		This method computes the objective function for λ-PPO, the basic implementation structure can be found in [1], 
		while the additional implementation tricks can be found in [2] and [3].
		The final update rules will be:
			Loss = λ_r * reward - sum_i( λ_i * cost_i ) 			> where λ_r = 1-sum_(λ_i)

		[1] Benchmarking Safe Exploration in Deep Reinforcement Learning. Achiam et al., 2019
		[2] Direct behavior specification via constrained reinforcement learning. Roy et al. 2021
		[3] Constrained reinforcement learning for robotics via scenario-based programming. Corsi et al. 2022

		NB: all the operation are performed in-line for simplicity, however the computations are performed in 
		parallel for the whole given batch
	"""
	def get_policy_loss_and_kl( self, mb_states, mb_actions, mb_log_probs, mb_advantages, mb_cost_advantages ):

		# Propagation of the state through the neural network to obtain the action probabilities, notice
		# that this step must conserve the gradient information beacuse the gradient will be computed from
		# the result of this operation, the other computation constitutes the target of the optimization
		_, new_logprob, entropy = self.actor.get_action( mb_states, action=mb_actions  )

		# Exploting logarithms properties to compute the ratio
		log_ratio = new_logprob - mb_log_probs 
		ratio = log_ratio.exp()

		# Commpute the KL-Divergence, approximated technique from [1] 
		# [1] http://joschu.net/blog/kl-approx.html
		with torch.no_grad(): approx_kl = ((ratio - 1) - log_ratio).mean()

		# Stadradization of the advantages, this follows the optimization trick from [1]
		# [1] https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
		mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
		    
		# Computation of the basic loss for the policy network (reward), this follows the standard
		# ppo-clip implementation from the original paper [1]
		# [1] Proximal Policy Optimization Algorithms, Schulman et al., 2017
		pg_rwloss1 = mb_advantages * ratio
		pg_rwloss2 = mb_advantages * torch.clamp(ratio, 0.8, 1.2)
		pg_rwloss = torch.min(pg_rwloss1, pg_rwloss2).mean()

		# Adding the entropy to improve the exploration, this is an implementation tricks from [1]
		# that follows the intuition from [2]. Here we also compute the reward multiplier following [3] and [4].
		# The reward multiplier is guaranteed to be in [min_lambda_reward, 1] by 'STEP 1'
		# [1] https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
		# [2] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al., 2018
		# [3] Direct behavior specification via constrained reinforcement learning. Roy et al. 2021
		# [4] Constrained reinforcement learning for robotics via scenario-based programming. Corsi et al. 2022
		lambdas_sum = numpy.array([lag.detach().numpy() for lag in self.lagrangian_multipliers]).sum()
		reward_multiplier = 1-lambdas_sum
		pg_loss = (reward_multiplier * pg_rwloss) + (self.args.entropy_coeff * entropy.mean())

		# Computation of the additional loss function for the costs function, this implements the constrained DRL
		for i in range( len(mb_cost_advantages) ):

			# Standardization of the costs advantages, this follows the optimization trick from [1], notice that 
			# the noramlization does not inclue the "/(mb_cost_advantages.std() + 1e-8)", this is an optimization from [2]
			# [1] https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
			# [2] Benchmarking Safe Exploration in Deep Reinforcement Learning. Achiam et al., 2019
			mb_cost_advantages_temp = (mb_cost_advantages[i] - mb_cost_advantages[i].mean())

			# Loss of the policy network (cost), the process is the same as the standard PPO, but the 
			# advantage is the cost-advantage
			pg_closs1 = mb_cost_advantages_temp * ratio
			pg_closs2 = mb_cost_advantages_temp * torch.clamp(ratio, 0.8, 1.2)
			pg_closs = torch.min(pg_closs1, pg_closs2).mean()

			# Update the objective function with the contribution of the cost loss functions, weighted
			# by the lagrangian multipliers
			pg_loss -= self.lagrangian_multipliers[i] * pg_closs

		return -pg_loss, approx_kl
	

	"""
		Computation of the loss for the value function, it implements MSE
	"""
	def get_value_loss( self, network, mb_states, mb_target ):
		predicted_value = network.forward( mb_states ).view(-1)
		v_loss = 0.5 * ((predicted_value - mb_target) ** 2).mean()
		return v_loss
	

	"""
		Method that compute the loss for the lagrangian multipliers updated with gradient descent, it
		implements 
						∇_(λ_k) Loss = -(C_k-d_k) where d_k is the threshold	(1)
		A clear derivation of this formula can be found in [1]

		[1] Constrained reinforcement learning for robotics via scenario-based programming. Corsi et al. 2022
	"""
	def update_lambda( self, avg_epcost ):

		# Iterate over all the possible cost functions (notice that our approach supports
		# multiple cost functions)
		for id, epcost in enumerate(avg_epcost):

			# Implements Eq. 1 to compute the loss and perform the gradient descent step
			# following the standard procedure from torch
			lambda_loss = -self.lagrangian_multipliers[id] * (epcost - self.args.cost_limit)
			self.lambda_optimizers[id].zero_grad()
			lambda_loss.backward()
			self.lambda_optimizers[id].step()
	
		# Return the actual loss function only for debugging purposes
		return lambda_loss


	"""
		Method that actually perform the gradient descent step following the
		torch implementation
	"""
	def optimization_step( self, network, optimizer, loss ):
		optimizer.zero_grad()
		loss.backward()		
		torch.nn.utils.clip_grad_norm_( network.parameters(), self.args.max_grad_norm )
		optimizer.step()


	"""
		Method that computes the softmax values for each sets of scores in x, we
		add also a 'temperature' paramter to balance the weights of the lagrangian multipliers
	"""
	def softmax( self, x, temperature=1, total=1):
		return (numpy.exp(x/temperature) / numpy.sum(numpy.exp(x/temperature), axis=0)) * total
		