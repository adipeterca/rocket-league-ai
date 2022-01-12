from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z
import numpy as np

class InitialStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
    
        # Set up our desired spawn location and orientation.
        desired_car_pos = [0,0,500] #x, y, z
        
        # Loop over every car in the game.
        for car in state_wrapper.cars:
            pos = desired_car_pos
            car.set_pos(*pos)
            car.boost = 1.0
