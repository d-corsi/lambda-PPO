import argparse
from scripts.lambda_ppo import LambdaPPO
from scripts.welcome import print_welcome_message


#TODO: implement the cost normalization
#TODO: test the cost function normalization
#
# class NormalizeCost( gymnasium.wrappers.NormalizeReward ):


""" 
	Main function that prints the welcome message and runs the training process
"""
def training( args ):
	print_welcome_message( args )
	algo = LambdaPPO( args )
	algo.main_loop()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Basic Training Settings
	parser.add_argument("--env-id", type=str, required=True)
	parser.add_argument("--num-costs", type=int, default=1)
	parser.add_argument("--verbose", type=int, default=0)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--updates", type=int, default=500)
	parser.add_argument("--num-envs", type=int, default=1)
	parser.add_argument("--render-mode", type=str, default="rgb_array")
	
	# PPO Hyperparams
	parser.add_argument("--num-steps", type=int, default=4096)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--max-grad-norm", type=float, default=0.5)
	parser.add_argument("--target-kl", type=float, default=0.02)
	parser.add_argument("--entropy-coeff", type=float, default=0.0)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-number", type=int, default=16)
	parser.add_argument("--value-epochs", type=int, default=60)
	parser.add_argument("--value-batch-number", type=int, default=32)

	# lambda-PPO Hyperparams
	parser.add_argument("--cost-limit", type=float, default=-1)
	parser.add_argument("--lambda-batch-number", type=int, default=2)
	parser.add_argument("--lambda-learning-rate", type=float, default=0.01)

	# Corsi et al. flags
	parser.add_argument("--min-lambda-reward", type=float, default=0.0)
	parser.add_argument("--start-train-lambda", type=float, default=0.0)
	parser.add_argument("--lambda-init", type=float, default=0.0)

	# Parse the arguments and run the training
	args = parser.parse_args()

	# -1 is special param for cost-limit, it means that we never start the training
	# of the lagrangian paramters; we obtain a naive PPO
	if args.cost_limit == -1: args.start_train_lambda = 1 

	# Adjust additional paramters
	args.num_steps = args.num_steps // args.num_envs
	args.start_train_lambda = int(args.start_train_lambda * args.updates)

	# Calling the main function to start the training
	training( args )

