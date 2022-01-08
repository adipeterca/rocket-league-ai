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

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        # By default we will chase the ball, but target_location can be changed later
        # target_location = ball_location

        # Draw a circle of locations
        target_points = []
        r = 100.0
        self.renderer.begin_rendering()
        #  for i in range(0, 2 * math.pi, 0.1):
            # self.renderer.draw_line_3d(car_location, Vec3(r * math.cos(i), 200, r * math.sin(i)), self.renderer.white())
        self.draw_points_to_fly(car_location)
        self.renderer.end_rendering()

        # Draw some things to help understand what the bot is thinking
        # self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        # self.renderer.draw_string_3d(car_location, 1, 1, f'Speed: {car_velocity.length():.1f}', self.renderer.white())
        # self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)

        # if self.start == 0:
        #     self.start_flying(packet)
        #     self.start = 1
        # else:
        #     self.fly(packet)
        #     packet.game_cars[0].boost=100
        controls = SimpleControllerState()

        # controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0

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
    
    def begin_front_flip(self, packet):
        # Send some quickchat just for fun
        self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.8, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)

    def draw_points_to_fly(self, car_location):
        """
        Function for displaying a circle of points on a fixed height and lines that connect the PLAYER to each one of them.
        Does not call begin_renderer() or end_renderer().

        :param car_location: the location of the car
        :author: Adi
        :mention: In progress!
        """
        # Default unit of measurement (about a car's length)
        unit = 500

        # The center of the circle
        circle_center = Vec3(0, 0, 3 * unit)

        # Draw some points that should be around the circle center

        self.renderer.draw_line_3d(car_location, circle_center + Vec3(unit, 0, 0), self.renderer.white())
        self.renderer.draw_line_3d(car_location, circle_center + Vec3(-unit, 0, 0), self.renderer.white())
        self.renderer.draw_line_3d(car_location, circle_center + Vec3(0, unit, 0), self.renderer.white())
        self.renderer.draw_line_3d(car_location, circle_center + Vec3(0, -unit, 0), self.renderer.white())


def draw_debug(renderer, car, target):
    renderer.begin_rendering()
    # draw a line from the car to the target
    renderer.draw_line_3d(car.physics.location, target, renderer.white())

    renderer.end_rendering()