from enum import Enum
import pygame
import time

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
RENDER_BORDER = 10
FRAME_RATE = 144

class StartingPos(Enum):
    BOTTOM = 0
    TOP = 1
    LEFT = 2
    RIGHT = 3
    
class Paths(Enum):
    TOP_BOTTOM = 0
    BOTTOM_TOP = 1

MAX_VELOCITY = 50
ACCELERATION = 8

    