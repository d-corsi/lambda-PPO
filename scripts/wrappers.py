import gymnasium

class NormalizeCost( gymnasium.wrappers.NormalizeReward ):
	
	def step(self, action):
		obs, rews, terminateds, truncateds, infos = self.env.step(action)
		infos["original_cost"] = infos["cost"]
		cost = [infos["cost"]]		
		self.returns = self.returns * self.gamma * (1 - terminateds) + cost
		cost = self.normalize(cost)
		infos["cost"] = cost[0]
		return obs, rews, terminateds, truncateds, infos
	

class MultiCostWrapper( gymnasium.Wrapper ):
	
	def __init__ ( self, env ):
		super().__init__(env)

		self.reward_monitor = 0
		self.cost_monitor = None


	def reset( self, **kwargs ):
		obs, info = self.env.reset( **kwargs )
		self.reward_monitor = 0
		self.cost_monitor = None

		return obs, info
		

	def step( self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)

		# Fix for single cost 
		if type(info["cost"]) is not list: info["cost"] = [info["cost"], 0, 0]
		if self.cost_monitor is None: self.cost_monitor = [0.0 for _ in info["cost"]]

		self.reward_monitor += reward
		for idx, c in enumerate(info["cost"]): self.cost_monitor[idx] += c

		if terminated or truncated:

			info['custom_info'] = {
				'tot_reward': self.reward_monitor,
				'tot_costs': self.cost_monitor
			}

		return obs, reward, terminated, truncated, info