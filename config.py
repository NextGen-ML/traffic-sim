from enum import Enum
import pygame
import time

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

MAX_VELOCITY = 10
ACCELERATION = 5
COLLISION_DISTANCE = 50 # def too large
WAIT_TIME = 10
DISTANCE_BETWEEN_CARS = 20


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