from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from Point import Point2D
import math

class CollidedTerminalCondition(TerminalCondition):
    def __init__(self):
        super().__init__()
        self.x_neg_limit = -4096.0
        self.x_pos_limit = 4096.0
        self.y_neg_limit = -5120.0 
        self.y_pos_limit = 5120.0
        self.z_floor_limit = 0.0
        self.z_ceil_limit = 2044.0  
        self.distance_considered_collided = 150
        self.corners_line = [
            [Point2D(2944.0, self.y_neg_limit), Point2D(self.x_pos_limit, -3968.0)],
            [Point2D(self.x_pos_limit, 3968.0), Point2D(2944.0 , self.y_pos_limit)],
            [Point2D(-2944.0, self.y_pos_limit), Point2D(self.x_neg_limit, 3968.0)],
            [Point2D(self.x_neg_limit, -3968.0), Point2D(-2944.0, self.y_neg_limit)]
        ]
    def reset(self, initial_state: GameState):
        # TODO Puts the car in the initial position. Consider initial position (0,0,300)
        # initial_state.players[0].car_data.position = [0,0, 300]
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        # We consider that a state is terminal if we collided with one of the 
        # arena boundries. We will compute the collision relating to the fact that the distance
        # till the one boundry is lower than 5.
        car_position = current_state.players[0].car_data.position
        x = car_position[0]
        y = car_position[1]
        z = car_position[2]
        # Verify that we are not close to a corners (The 2D Layout of the map is not a square)
        P0 = Point2D(x, y)
        for line in self.corners_line:
            P1 = Point2D(line[0].x, line[0].y)
            P2 = Point2D(line[1].x, line[1].y)
            if self.distance(P1, P2, P0) <= self.distance_considered_collided:

                return True
        # If we are not close to the corners, we must verify that we are not close to a map boundry on each axis
        if x > 0:
            if self.x_pos_limit - x <= self.distance_considered_collided:
                return True
        else:
            if abs(self.x_neg_limit - x) <= self.distance_considered_collided:
                return True
        if y > 0:
            if self.y_pos_limit - y <= self.distance_considered_collided:
                return True
        else:
            if abs(self.y_neg_limit - y) <= self.distance_considered_collided:
                return True
        if z <= self.distance_considered_collided or self.z_ceil_limit - z <= self.distance_considered_collided:
            return True

        return False

    def distance(self, P1, P2, P0):
        return abs((P2.x - P1.x)*(P1.y - P0.y)-(P1.x-P0.x)*(P2.y-P1.y)) / math.sqrt((P2.x-P1.x)*(P2.x-P1.x)+(P2.y-P1.y)*(P2.y-P1.y)) 