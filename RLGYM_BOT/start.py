import rlgym
from AirStateSetter import InitialStateSetter
from Reward import AirReward
#from rlgym.utils.reward_functions.common_rewards import ConditionalRewardFunction
from rlgym.utils.reward_functions import RewardFunction
from TouchArenaBoundariesTerminalCondition import CollidedTerminalCondition
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 110

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

env = rlgym.make(game_speed=100 ,state_setter=InitialStateSetter(), reward_fn = AirReward(), terminal_conditions=[CollidedTerminalCondition(), TimeoutCondition(max_steps)] )

while True:
    obs = env.reset()
    done = False

    while not done:
      #Here we sample a random action. If you have an agent, you would get an action from it here.
      action = env.action_space.sample() 
      
      next_obs, reward, done, gameinfo = env.step(action)
      
      obs = next_obs