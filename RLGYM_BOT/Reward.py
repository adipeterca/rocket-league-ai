from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from TouchArenaBoundariesTerminalCondition import CollidedTerminalCondition
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
import timeit
import copy


class AirReward(RewardFunction):
	start_time = timeit.default_timer()
	last_negative_reward = ""
	collided_obj = CollidedTerminalCondition()
	def reset(self, initial_state: GameState):
		pass

	def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
		end_time = timeit.default_timer()
		time_passed = end_time - AirReward.start_time
		if time_passed >= 0.01 and AirReward.collided_obj.is_terminal(state) == False:
			# print(time_passed)
			AirReward.start_time = copy.deepcopy(end_time)
			return 0.0015 * time_passed
		return 0.0

	def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
		current_time_out = timeit.default_timer()
		if AirReward.collided_obj.is_terminal(state) == True:
			# print("NEGATIVE REWARD")
			time_between = 0.03
			if AirReward.last_negative_reward != "":
				time_between = current_time_out - AirReward.last_negative_reward
			AirReward.last_negative_reward = copy.deepcopy(current_time_out)
			return -0.1 / time_between
		if type(AirReward.last_negative_reward) is float and (abs(current_time_out - AirReward.last_negative_reward) >= 0.6):
			# print("======================================\nWON!======================================\n")
			file_object = open('D:\timePassed.txt', 'a')
			file_object.write("======================================\nWON!======================================\n")
			file_object.close()
			return 10.0

		return 0.0
