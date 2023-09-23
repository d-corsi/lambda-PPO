CYAN_COL = '\033[96m'
BLUE_COL = '\033[94m'
RED_COL = '\033[91m'
GREEN_COL = '\033[92m'
YELLOW_COL = '\033[93m'
RESET_COL = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def print_welcome_message( args ):
	
	print( f"\n{RED_COL}{BOLD}=====================================================================" )
	print( f"Welcome to λ-PPO, an optimized version of the standard Lagrangian-PPO" )
	print( f"====================================================================={RESET_COL}\n" )
	
	print( f"You are running the training on {GREEN_COL}{args.env_id}{RESET_COL} \
for {GREEN_COL}{args.updates} episodes{RESET_COL}." )
	print( f"The number of {GREEN_COL}parallel environments is {args.num_envs}{RESET_COL}, each of \
which will run for {GREEN_COL}{args.num_steps} steps{RESET_COL} before the \nupdate. Please \
ensure that the selected enviornment can run for more steps before truncation (for \nexample \
{UNDERLINE}'circle' requires at least 500 steps while 'goal', 'push' and 'locomotion' requires 1000 steps{RESET_COL})." )
	
	print( f"\nYou indicate that your environment returns {CYAN_COL}{args.num_costs} cost \
functions{RESET_COL} in the 'info' dictionary, \nplease check this value carefully. \
The {CYAN_COL}cost threshold is {args.cost_limit}{RESET_COL} (remember that -1 is a key \nvalue for 'no limit'). \
The process of training the lagrangian multipliers starts afer {CYAN_COL}{args.start_train_lambda} updates.{RESET_COL}" )
	
	print( f"\nThe {YELLOW_COL}verbose level is {args.verbose}{RESET_COL} (where 0 means 'only print' and 1 \
'save in WandB'). All the other \nhyperparamters can be found in the main file (i.e., 'main.py').")
	
	print( f"\n{RED_COL}{BOLD}======================================" )
	print( f"The training process will now start..." )
	print( f"======================================{RESET_COL}\n" )