import numpy as np
import gym
from gym import spaces
from config import Config
from sim import run_simulation, save_and_plot_data, count_collisions, reward_records
from intersection import Intersection, Paths, StartingPos

class IntersectionEnv(gym.Env):
    def __init__(self, config, intersection, agent=None):
        super(IntersectionEnv, self).__init__()
        self.config = config
        self.intersection = intersection
        self.agent = agent
        self.bottom_top_interval = 75
        self.left_right_interval = 75
        self.bottom_top_next_interval = self.bottom_top_interval + np.random.randint(-10, 10)
        self.left_right_next_interval = self.left_right_interval + np.random.randint(-10, 10)

        self.action_space = spaces.Box(
            low=np.array([50, 25, 10, 5, 10]),
            high=np.array([200, 100, 35, 15, 30]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )
        self.collision_records = []
        self.intersection_records = []
        self.reward_records = []

    def reset(self):
        self.collision_records = []
        self.intersection_records = []
        self.reward_records = []
        # Instead of calling update_parameters() without arguments,
        # let's reset to default values or use the current values
        self.config.update_parameters(
            max_velocity=self.config.MAX_VELOCITY,
            acceleration=self.config.ACCELERATION,
            collision_distance=self.config.COLLISION_DISTANCE,
            wait_time=self.config.WAIT_TIME,
            distance_between_cars=self.config.DISTANCE_BETWEEN_CARS
        )
        self.bottom_top_interval = 50
        self.left_right_interval = 50
        return self._get_state(is_first_interval=True)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        max_velocity, acceleration, collision_distance, wait_time, distance_between_cars = action
        self.config.update_parameters(
            max_velocity=max_velocity,
            acceleration=acceleration,
            collision_distance=collision_distance,
            wait_time=wait_time,
            distance_between_cars=distance_between_cars
        )

        interval_results, self.bottom_top_next_interval, self.left_right_next_interval = run_simulation(self.config, self.agent)
        
        total_crossings = 0
        total_collisions = 0
        for interval_collisions, interval_crossings, is_first_interval, bottom_top_next_interval, left_right_next_interval in interval_results:
            self.collision_records.append(interval_collisions)
            self.intersection_records.append(interval_crossings)
            total_crossings += interval_crossings
            total_collisions += interval_collisions

        reward = total_crossings - total_collisions * 100
        reward = max(min(reward, 200), -750)
        self.reward_records.append(reward)

        # Get the last interval's collision and crossing counts
        last_collisions = interval_results[-1][0] if interval_results else 0
        last_crossings = interval_results[-1][1] if interval_results else 0

        obs = self._get_state(is_first_interval, bottom_top_next_interval, left_right_next_interval, last_collisions, last_crossings)
        done = len(self.collision_records) >= 3

        return obs, reward, done, {}

    def _get_state(self, is_first_interval=False, bottom_top_next_interval=0, left_right_next_interval=0, last_collisions=0, last_crossings=0):
        state = np.array([
            bottom_top_next_interval,
            left_right_next_interval,
            1 if is_first_interval else 0,
            last_collisions,   
            last_crossings     
        ], dtype=np.float32)

        state_mean = np.mean(state)
        state_std = np.std(state) + 1e-8  # Avoid division by zero
        normalized_state = (state - state_mean) / state_std

        return normalized_state

    def render(self, mode='human'):
        self._sync_records_length()
        save_and_plot_data(self.collision_records, self.intersection_records, self.reward_records)

    def _sync_records_length(self):
        max_length = max(len(self.collision_records), len(self.intersection_records), len(self.reward_records))
        while len(self.collision_records) < max_length:
            self.collision_records.append(0)
        while len(self.intersection_records) < max_length:
            self.intersection_records.append(0)
        while len(self.reward_records) < max_length:
            self.reward_records.append(0)

# Define intersection
four_way = Intersection(
    motion_path_array=[Paths.TOP_BOTTOM, Paths.BOTTOM_TOP, Paths.LEFT_RIGHT, Paths.RIGHT_LEFT],
    number_of_roads=4,
    starting_positions=[StartingPos.BOTTOM, StartingPos.TOP, StartingPos.LEFT, StartingPos.RIGHT],
    size=(100, 100)
)

config = Config()
env = IntersectionEnv(config, four_way)