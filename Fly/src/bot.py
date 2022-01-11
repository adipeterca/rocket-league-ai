from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from rlbot.messages.flat.BoostOption import BoostOption

import math
import numpy as np
import random
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import activations

def build_model_boost():
    """
    Build a predefined model of a Neural Network (12, 50, 1).

    Important observation: the decision of whether or not to apply boost in a given state could also be implemented
    as a perceptron, as it is a liniar separable problem.

    As input, it takes 4 Vec3 parameters:
    - location
    - rotation
    - velocity
    - angular velocity
    As output, it gives 1 integer parameter:
    - boost (between 0 and 1)
    """
    model = Sequential()
    model.add(Dense(100, input_shape=(12,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

def build_model_direction():
    """
    Build a predefined model of a Neural Network (12, 100, 4).

    As input, it takes 4 Vec3 parameters:
    - location
    - rotation
    - velocity
    - angular velocity
    As output, it gives 4 integer parameters:
    - pitch up (between 0 and 1)
    - pitch down (between 0 and 1)
    - yaw left (between 0 and 1)
    - yaw right (between 0 and 1)

    :return: the neural network model
    """
    model = Sequential()
    model.add(Dense(100, input_shape=(12,), activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

def output_to_controls(nn_output_direction, nn_output_boost) -> SimpleControllerState:
    """
    Function for converting the given Neural Network outputs into a SimpleControllerState.

    :param nn_output_direction: output from the directional NN
    :param nn_output_boost: whether or not to apply the boost
    :return: a controller state
    """

def build_inputs(packet: GameTickPacket, car_index):
    """
    Function for converting the given Game Tick packet into viable Neural Network inputs.

    :param packet: the given game tick packet
    :param car_index: index of the current car
    :return: NN inputs 
    """
    inputs = []
    location = packet.game_cars[car_index].physics.location
    rotation = packet.game_cars[car_index].physics.rotation
    velocity = packet.game_cars[car_index].physics.velocity
    angular_velocity = packet.game_cars[car_index].physics.angular_velocity
    
    # Append the location values
    inputs.append(location.x)
    inputs.append(location.y)
    inputs.append(location.z)

    # Append the rotation values
    inputs.append(rotation.pitch)
    inputs.append(rotation.yaw)
    inputs.append(rotation.roll)

    # Append the velocity values
    inputs.append(velocity.x)
    inputs.append(velocity.y)
    inputs.append(velocity.z)

    # Append the angular_velocity
    inputs.append(angular_velocity.x)
    inputs.append(angular_velocity.y)
    inputs.append(angular_velocity.z)

    return np.array(inputs).reshape(12, 1).T

class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None

        # Variabile used to determine in which state the car is
        self.start = 0

        # Neural network models
        self.nn_direction = build_model_direction()
        self.nn_boost = build_model_boost()

        self.initial_car_location = None

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        # self.boost_pad_tracker.initialize_boosts(self.get_field_info())
        pass

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)

        # Debug rendering
        self.renderer.begin_rendering()
        self.draw_points_to_fly(car_location)
        self.renderer.end_rendering()

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # # # This is where the neural network decision should go
        # inputs = build_inputs(packet, self.index)

        # output_direction = self.nn_direction(inputs)
        # output_boost = self.nn_boost(inputs)

        # controls = output_to_controls(output_direction, output_boost)

        # Initial takeoff
        if self.initial_car_location == None:
            # Get the original car location
            self.initial_car_location = car_location

            # Take off
            self.start_flying(packet)
            self.start = 1
        else:
            if car_location.y > self.initial_car_location.y:
                # Balance the car as to go forward
                print("going forward")
                self.active_sequence = Sequence([
                ControlStep(duration=0.3, controls=SimpleControllerState(boost=True, roll=1.0)),
                ControlStep(duration=0.26, controls=SimpleControllerState(boost=False, roll=1.0)),
                ControlStep(duration=0.15, controls=SimpleControllerState(boost=True, roll=1.0)),
                ControlStep(duration=0.05, controls=SimpleControllerState(boost=False, pitch=-0.01, roll=1.0))
            ])
            else:
                # Balance the car as to go backwards
                print("going backwards")
                self.active_sequence = Sequence([
                ControlStep(duration=0.3, controls=SimpleControllerState(boost=True, roll=1.0)),
                ControlStep(duration=0.2, controls=SimpleControllerState(boost=False, roll=1.0)),
                ControlStep(duration=0.15, controls=SimpleControllerState(boost=True, roll=1.0)),
                ControlStep(duration=0.05, controls=SimpleControllerState(boost=False, pitch=0.01, roll=1.0))
            ])

        # Initial fly
        # if self.start == 0:
        #     self.start_flying(packet)
        #     self.start = 1
        # elif car_location.y > -1000.00: # Does not collide with the wall
        #     self.fly(packet)
        #     # packet.game_cars[0].boost = 100
        # else: 
        #     self.active_sequence = Sequence([
        #         ControlStep(duration=0.3, controls=SimpleControllerState(boost=True)),
        #         ControlStep(duration=0.16, controls=SimpleControllerState(boost=False)),
        #         ControlStep(duration=0.15, controls=SimpleControllerState(boost=True)),
        #         ControlStep(duration=0.1, controls=SimpleControllerState(boost=False, pitch=0.3))
        #     ])

        controls = SimpleControllerState()

        return controls

    def fly(self, packet):
        """
        Function for hardcoded flying.

        :param packet: the game tick packet for the current frame
        :return: an active sequence of actions
        """
        # self.active_sequence = Sequence([])
        # if self.start == 1: 
        #     self.active_sequence = Sequence([
        #         ControlStep(duration=1.5, controls=SimpleControllerState(boost=True)),
        #         ControlStep(duration=0.1, controls=SimpleControllerState(boost=False)),
        #     ])
        #     self.start = 2
        # else:
        pitch_rotation = 1.5 - packet.game_cars[0].physics.rotation.pitch
        self.active_sequence = Sequence([
            ControlStep(duration=0.3, controls=SimpleControllerState(boost=True)),
            ControlStep(duration=0.26, controls=SimpleControllerState(boost=False)),
            ControlStep(duration=0.15, controls=SimpleControllerState(boost=True, pitch=pitch_rotation))
        ])
        # 0.54, 0.5, 0.3  
        return self.active_sequence.tick(packet)

    def start_flying(self, packet):
        """
        Function for initiating the jump motion.

        :param packet: the game tick packet for the current frame
        :return: an active sequence of actions
        """
        self.active_sequence = Sequence([
            ControlStep(duration=5, controls=SimpleControllerState()),
            ControlStep(duration=0.5, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.5, controls=SimpleControllerState(jump=False, pitch=0.51)),
            ControlStep(duration=0.5, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=1.5, controls=SimpleControllerState(boost=True)),
            ControlStep(duration=0.1, controls=SimpleControllerState(boost=False))
        ])
        return self.active_sequence.tick(packet)

    def draw_points_to_fly(self, car_location):
        """
        Function for displaying a circle of points on a fixed height and lines that connect the PLAYER to each one of them.
        Does not call begin_renderer() or end_renderer().

        :param car_location: the location of the car
        """
        # Default unit of measurement (about a car's length)
        unit = 500

        # Draw rectangles in a circular pattern using an angle that goes about the center of a circle
        # x position = unit * cos(angle)
        # y position = unit * sin(angle)
        # z position = unit (how high from the base of the field the rectangle should be)
        # angle is updated each iteration such that only 20 points are drawn
        angle = 0.0
        while angle < 2 * math.pi:
            self.renderer.draw_rect_3d(Vec3(unit * math.cos(angle), unit * math.sin(angle), 3 * unit), 20, 20, 0, self.renderer.green())
            angle += 0.314