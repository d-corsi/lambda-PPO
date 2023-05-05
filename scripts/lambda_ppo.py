from scripts.common import ReinforcementLearning
import numpy, torch

class LambdaPPO( ReinforcementLearning ):
    

	def __init__( self, args ):
		super().__init__( args )

	
	def _learningrate_annealing( self, update ):
		frac = 1.0 - (update - 1.0) / self.args.updates
		lrnow = frac * self.args.learning_rate
		self.actor_optimizer.param_groups[0]["lr"] = lrnow
		self.critic_optimizer.param_groups[0]["lr"] = lrnow
		self.cost_critic_optimizer.param_groups[0]["lr"] = lrnow


	def _train_networks( self, memory_buffer, avg_reward=None, avg_cost=None ):

		# Preprocessing of the data
		memory_buffer.add_GAE_reward( self.args, self.critic )
		memory_buffer.add_GAE_cost( self.args, self.cost_critic )
		memory_buffer.add_target_return()
		memory_buffer.add_target_cost()
		memory_buffer.flatten_observation()

		# Extract the relevant infos from the buffer
		states, actions, log_probs, advantages, target_returns = memory_buffer.get_update_data()
		cost_advantages, cost_target_returns = memory_buffer.get_cost_update_data()

		#
		buffer_indices = numpy.arange( self.args.num_steps * self.args.num_envs )

		if self.current_update > self.args.start_train_lambda:

			# Out-of-the-box update for the lagrangian multiplier
			cost_batches = numpy.array_split(avg_cost, min(self.args.lambda_batch_number, len(avg_cost)))
			cost_batch = [numpy.mean(b) for b in cost_batches]

			for cost in cost_batch: self.update_lambda(cost)

			# Lambda Clamp if necessary:
			if self.args.test_2 == 0:
				# Softplus, no normalization required
				pass
			elif self.args.test_2 == 1:
				# Hard constraint >= 0
				self.lagrangian_multiplier.data.clamp_(min=0)
			elif self.args.test_2 == 2:
				# Corsi et al.
				self.lagrangian_multiplier.data.clamp_(min=0, max=(1-self.args.min_lambda_reward))
			else:
				# Invalid Setting 
				raise ValueError
							

		# Updates of the value functions
		for _ in range( self.args.value_epochs ):

			#
			numpy.random.shuffle( buffer_indices )
			batches = numpy.array_split( buffer_indices, self.args.value_batch_number )

			for minibatch in batches:

				# Comptutation of the loss functions
				value_loss = self.get_value_loss( self.critic, states[minibatch], target_returns[minibatch] )
				cost_value_loss = self.get_value_loss( self.cost_critic, states[minibatch], cost_target_returns[minibatch] )

				# Perform the actual optimization step
				self.optimization_step( self.critic, self.critic_optimizer, value_loss )
				self.optimization_step( self.cost_critic, self.cost_critic_optimizer, cost_value_loss )


		# Updates of the policy networks
		for vep in range( self.args.epochs ):

			#
			numpy.random.shuffle( buffer_indices )
			batches = numpy.array_split( buffer_indices, self.args.batch_number )

			for minibatch in batches:

				# Comptutation of the loss functions
				policy_loss, approx_kl = self.get_policy_loss_and_kl( states[minibatch], actions[minibatch], \
							 log_probs[minibatch], advantages[minibatch], cost_advantages[minibatch] )
				
				# Perform the actual optimization step
				self.optimization_step( self.actor, self.actor_optimizer, policy_loss )
				
			if (approx_kl.mean().cpu().detach().numpy()) > self.args.target_kl: break
			# 	# Just check the last kl-divergence, supposed to be the higher
			# 	print( f"\tDEBUG early stop after iteration {vep}/{self.args.epochs} for kl-divergence" )
			# 	break
				
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


	def get_policy_loss_and_kl( self, mb_states, mb_actions, mb_log_probs, mb_advantages, mb_cost_advantages ):
		_, new_logprob, entropy = self.actor.get_action( mb_states, action=mb_actions  )
		log_ratio = new_logprob - mb_log_probs # Exploting logarithms properties to compute the ratio
		ratio = log_ratio.exp()

		# Commpute the KL-Divergence [http://joschu.net/blog/kl-approx.html]
		with torch.no_grad(): approx_kl = ((ratio - 1) - log_ratio).mean()

		if self.args.test_0 == 0:	
			# Naive Normalization
			pass
		elif self.args.test_0 == 1:	
			# Standardization
			mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
			mb_cost_advantages = (mb_cost_advantages - mb_cost_advantages.mean()) #/ (mb_cost_advantages.std() + 1e-8)
		elif self.args.test_0 == 2:	
			# Normalization
			mb_advantages = (mb_advantages - mb_advantages.min()) / (mb_advantages.max() - mb_advantages.min())
			mb_cost_advantages = (mb_cost_advantages - mb_cost_advantages.min()) / (mb_cost_advantages.max() - mb_cost_advantages.min())
		else:
			# Invalid Setting 
			raise ValueError
									    
		# Loss of the policy network (reward)
		pg_rwloss1 = mb_advantages * ratio
		pg_rwloss2 = mb_advantages * torch.clamp(ratio, 0.8, 1.2)
		pg_rwloss = torch.min(pg_rwloss1, pg_rwloss2).mean()

		# Loss of the policy network (reward)
		pg_closs1 = mb_cost_advantages * ratio
		pg_closs2 = mb_cost_advantages * torch.clamp(ratio, 0.8, 1.2)
		pg_closs = torch.min(pg_closs1, pg_closs2).mean()

		if self.current_update <= self.args.start_train_lambda: 
			# print( "DEBUG: ignore cost penalty before start-train-lambda")
			pg_closs = 0

		reward_multiplier = 1
		if self.args.test_2 == 2: reward_multiplier = (1 - self.lagrangian_multiplier.detach().numpy())

		if self.args.test_2 == 0:
			pg_loss = reward_multiplier * pg_rwloss - torch.nn.Softplus()(self.lagrangian_multiplier) * pg_closs + self.args.entropy_coeff * entropy.mean() 
		else:
			pg_loss = reward_multiplier * pg_rwloss - self.lagrangian_multiplier * pg_closs + self.args.entropy_coeff * entropy.mean() 
		
		return -pg_loss, approx_kl
	

	def get_value_loss( self, network, mb_states, mb_target ):
		predicted_value = network.forward( mb_states ).view(-1)
		v_loss = 0.5 * ((predicted_value - mb_target) ** 2).mean()
		return v_loss
	

	def update_lambda( self, avg_epcost ):

		if self.args.test_1 == 2: 
			self._no_grad_lambda_update( avg_epcost )
			return 0

		if self.args.test_1 == 3:
			proportional = (avg_epcost - self.args.cost_limit)
			self.args.PID_integrative = max(self.args.PID_integrative + proportional, 0)
			derivative = max(avg_epcost - self.args.PID_old_cost, 0)

			Kp, Ki, Kd = 0.1, 0.0001, 0.5
			new_value = ( Kp * proportional + Ki * self.args.PID_integrative + Kd * derivative )
			self.lagrangian_multiplier = torch.nn.Parameter( torch.as_tensor(new_value), requires_grad=False )

			return 0

		#lambda_loss = -self.lagrangian_multiplier * torch.Tensor(avg_epcost - self.args.cost_limit).to(self.device)
		lambda_loss = -self.lagrangian_multiplier * (avg_epcost - self.args.cost_limit)
		self.lambda_optimizer.zero_grad()
		lambda_loss.backward()
		self.lambda_optimizer.step()
	
		return lambda_loss
	

	def _no_grad_lambda_update( self, avg_epcost ):
		if avg_epcost > self.args.cost_limit: self.lagrangian_multiplier += 0.01
		else: self.lagrangian_multiplier -= 0.01


	def optimization_step( self, network, optimizer, loss ):
		optimizer.zero_grad()
		loss.backward()		
		torch.nn.utils.clip_grad_norm_( network.parameters(), self.args.max_grad_norm )
		optimizer.step()
		