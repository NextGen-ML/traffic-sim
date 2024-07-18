from typing import List
from config import *

class Intersection:
    def __init__(self, motion_path_array, number_of_roads, starting_positions, size):
        self.mp_array: List[Paths] = motion_path_array
        self.number_of_roads: int = number_of_roads
        self.start_array = List[StartingPos] = starting_positions
        self.size: tuple[int, int] = size
        
        
four_way = Intersection(
    motion_path_array=[Paths.TOP_BOTTOM, Paths.BOTTOM_TOP, Paths.LEFT_RIGHT, Paths.RIGHT_LEFT],
    number_of_roads=4,
    starting_positions=[StartingPos.BOTTOM, StartingPos.TOP, StartingPos.LEFT, StartingPos.RIGHT],
    size=(100, 100)
)
        