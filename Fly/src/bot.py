import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from rlbot.messages.flat.BoostOption import BoostOption

import random


class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.start = 0

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

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

        # Initial fly
        if self.start == 0:
            self.start_flying(packet)
            self.start = 1
        else:
            self.fly(packet)
            packet.game_cars[0].boost=100
        controls = SimpleControllerState()

        return controls

    def fly(self, packet):
        """
        Function for hardcoded flying.

        :author: Hutu Alexandru
        """
        self.active_sequence = Sequence([])
        if self.start == 1: 
            self.active_sequence = Sequence([
                ControlStep(duration=1.5, controls=SimpleControllerState(boost=True)),
                ControlStep(duration=0.5, controls=SimpleControllerState(boost=False)),
            ])
            self.start = 2
        else:
            pitch_rotation = 1.5 - packet.game_cars[0].physics.rotation.pitch  
            print(packet.game_cars[0].physics.rotation.pitch )
            print("======================================================")
            print(packet.game_cars[0].physics.location.x,packet.game_cars[0].physics.location.y,packet.game_cars[0].physics.location.z )
            print("======================================================")
            self.active_sequence = Sequence([
                ControlStep(duration=0.54, controls=SimpleControllerState(boost=True, roll=0.5)),
                ControlStep(duration=0.5, controls=SimpleControllerState(boost=False,roll=0.5)),
                ControlStep(duration=0.3, controls=SimpleControllerState(boost=True, pitch=pitch_rotation,roll=0.5))
            ])
                
        return self.active_sequence.tick(packet)

    def start_flying(self, packet):
        """
        Function for initiating the jump motion.

        :return: ??
        :author: Hutu Alexandru
        """
        self.active_sequence = Sequence([
            ControlStep(duration=5, controls=SimpleControllerState()),
            ControlStep(duration=0.5, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.5, controls=SimpleControllerState(jump=False, pitch=0.50)),
            ControlStep(duration=0.5, controls=SimpleControllerState(jump=True)),
        ])
        return self.active_sequence.tick(packet)

    def draw_points_to_fly(self, car_location):
        """
        Function for displaying a circle of points on a fixed height and lines that connect the PLAYER to each one of them.
        Does not call begin_renderer() or end_renderer().

        :param car_location: the location of the car
        :author: Adi & Theo
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