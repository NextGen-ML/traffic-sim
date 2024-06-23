from enum import Enum
import pygame
import time
from car_queue import CarQueue

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