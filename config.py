from enum import Enum
import pygame
import time

# Constants
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
RENDER_BORDER = 10
FRAME_RATE = 60

class StartingPos(Enum):
    BOTTOM = 0
    TOP = 1
    LEFT = 2
    RIGHT = 3
    
class Paths(Enum):
    TOP_BOTTOM = 0
    BOTTOM_TOP = 1
    LEFT_RIGHT = 2
    RIGHT_LEFT = 3
    
def is_partner_path(path1, path2):
    if path1 in [Paths.TOP_BOTTOM, Paths.BOTTOM_TOP] and path2 in [Paths.TOP_BOTTOM, Paths.BOTTOM_TOP]:
        return True
    elif path1 in [Paths.LEFT_RIGHT, Paths.RIGHT_LEFT] and path2 in [Paths.LEFT_RIGHT, Paths.RIGHT_LEFT]:
        return True
    
    return False

def get_partner_path(path):
    if path == Paths.TOP_BOTTOM:
        return Paths.BOTTOM_TOP
    elif path == Paths.BOTTOM_TOP:
        return Paths.TOP_BOTTOM
    elif path == Paths.LEFT_RIGHT:
        return Paths.RIGHT_LEFT
    elif path == Paths.RIGHT_LEFT:
        return Paths.LEFT_RIGHT
    
    return None
    


class Config:
    def __init__(self):
        self.MAX_VELOCITY = 100
        self.ACCELERATION = 50
        self.COLLISION_DISTANCE = 50
        self.WAIT_TIME = 5
        self.DISTANCE_BETWEEN_CARS = 20

    # For changing the parameters using our reinforcement learning model
    def update_parameters(self, max_velocity=None, acceleration=None, collision_distance=None, wait_time=None, distance_between_cars=None):
        if max_velocity is not None:
            self.MAX_VELOCITY = max_velocity
        if acceleration is not None:
            self.ACCELERATION = acceleration
        if collision_distance is not None:
            self.COLLISION_DISTANCE = collision_distance
        if wait_time is not None:
            self.WAIT_TIME = wait_time
        if distance_between_cars is not None:
            self.DISTANCE_BETWEEN_CARS = distance_between_cars

    def get_parameters(self):
        return {
            'MAX_VELOCITY': self.MAX_VELOCITY,
            'ACCELERATION': self.ACCELERATION,
            'COLLISION_DISTANCE': self.COLLISION_DISTANCE,
            'WAIT_TIME': self.WAIT_TIME,
            'DISTANCE_BETWEEN_CARS': self.DISTANCE_BETWEEN_CARS
        }