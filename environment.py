import numpy as np
import gym
from gym import spaces
from config import Config
from sim import run_simulation, save_and_plot_data, count_collisions, bottom_top_next_interval, left_right_next_interval
from intersection import Intersection, Paths, StartingPos

class IntersectionEnv(gym.Env):
    def __init__(self, config, intersection):
        super(IntersectionEnv, self).__init__()
        self.config = config
        self.intersection = intersection
        self.bottom_top_interval = 50
        self.left_right_interval = 50

        self.action_space = spaces.Box(
            low=np.array([50, 25, 20, 2, 5]),  # min values for velocity, acceleration, collision distance, wait time, and distance between cars
            high=np.array([250, 150, 120, 10, 75]),  # max values for the same parameters
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(8,),  
            dtype=np.float32
        )
        self.collision_records = []
        self.intersection_records = []

    def reset(self):
        self.collision_records = []
        self.intersection_records = []
        self.config.update_parameters()
        self.bottom_top_interval = 50
        self.left_right_interval = 50
        return self._get_state()

    def step(self, action):
        max_velocity, acceleration, collision_distance, wait_time, distance_between_cars = action
        self.config.update_parameters(
            max_velocity=max_velocity,
            acceleration=acceleration,
            collision_distance=collision_distance,
            wait_time=wait_time,
            distance_between_cars=distance_between_cars
        )
        
        interval_results = run_simulation()
        total_crossings = sum(crossings for _, crossings in interval_results)
        total_collisions = sum(collisions for collisions, _ in interval_results)

        for interval_collisions, interval_crossings in interval_results:
            self.collision_records.append(interval_collisions)
            self.intersection_records.append(interval_crossings)

        reward = total_crossings - total_collisions * 10
        obs = self._get_state()
        done = len(self.collision_records) >= 10

        return obs, reward, done, {}

    def _get_state(self):
        return np.array([
            self.collision_records[-1] if self.collision_records else 0,
            self.intersection_records[-1] if self.intersection_records else 0,
            bottom_top_next_interval,
            left_right_next_interval,
            len(self.intersection.motion_path_array),
            self.intersection.number_of_roads,
            self.intersection.size[0],  # width of the intersection
            self.intersection.size[1],  # height of the intersection
        ], dtype=np.float32)

    def render(self, mode='human'):
        save_and_plot_data(self.collision_records, self.intersection_records)

# Define intersection
four_way = Intersection(
    motion_path_array=[Paths.TOP_BOTTOM, Paths.BOTTOM_TOP, Paths.LEFT_RIGHT, Paths.RIGHT_LEFT],
    number_of_roads=4,
    starting_positions=[StartingPos.BOTTOM, StartingPos.TOP, StartingPos.LEFT, StartingPos.RIGHT],
    size=(100, 100)
)

# Initialize environment with config and intersection
config = Config()
env = IntersectionEnv(config, four_way)