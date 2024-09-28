import pygame
import pandas as pd
import matplotlib.pyplot as plt
from car import Car
from config import *
from random import randint
import time
from collections import defaultdict
import numpy as np

# --- CONSTANTS ---
MAX_CARS_PER_DIRECTION = 5
MAX_TOTAL_CARS = 8

# --- SIMULATION WINDOW ---

def initialize_window():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Car Simulation')
    clock = pygame.time.Clock()
    return screen, clock

def update_screen(screen, total_crossings, collisions, interval_count, config):
    screen.fill((255, 255, 255))
    rect_width, rect_height = 100, 100
    rect_x = (SCREEN_WIDTH / 2) - (rect_width / 2)
    rect_y = (SCREEN_HEIGHT / 2) - (rect_height / 2)
    pygame.draw.rect(screen, (0, 0, 0), (rect_x, rect_y, rect_width, rect_height), width=5)

    font = pygame.font.Font(None, 16)
    collision_font = pygame.font.Font(None, 26)
    interval_font = pygame.font.Font(None, 26)

    params = config.get_parameters()
    params_text = [
        f"MAX_VELOCITY: {params['MAX_VELOCITY']}",
        f"ACCELERATION: {params['ACCELERATION']}",
        f"COLLISION_DISTANCE: {params['COLLISION_DISTANCE']}",
    ]

    for idx, text in enumerate(params_text):
        param_surface = font.render(text, True, (0, 0, 0))
        screen.blit(param_surface, (10, 10 + idx * 20))

    collisions_text = collision_font.render(f"Collisions: {count_collisions(collisions)}", True, (255, 0, 0))
    screen.blit(collisions_text, (340, 10))

    crossings_text = collision_font.render(f"Crossings: {total_crossings}", True, (0, 0, 255))
    screen.blit(crossings_text, (340, 35))

    interval_text = interval_font.render(f"{interval_count}", True, (80, 80, 80))
    screen.blit(interval_text, (20, SCREEN_HEIGHT - 50))

def handle_events(cars):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_k and cars:
                temp = not cars[0].get_row() if cars[0].get_row() is not None else False
                cars[0].set_row(temp)
    return True

# --- CAR BEHAVIOR ---

def is_close_to(x1, y1, x2, y2, tolerance):
    dist = abs((((x2-x1)**2) + ((y2-y1)**2)) ** 0.5)
    return tolerance > dist

def return_car(path, config):
    if path == Paths.TOP_BOTTOM:
        return Car(StartingPos.TOP, Paths.TOP_BOTTOM, randint(0, 100000), config)
    elif path == Paths.BOTTOM_TOP:
        return Car(StartingPos.BOTTOM, Paths.BOTTOM_TOP, randint(0, 100000), config)
    elif path == Paths.LEFT_RIGHT:
        return Car(StartingPos.LEFT, Paths.LEFT_RIGHT, randint(0, 100000), config)
    elif path == Paths.RIGHT_LEFT:
        return Car(StartingPos.RIGHT, Paths.RIGHT_LEFT, randint(0, 100000), config)
    return None

def add_collision(car1, car2, collisions, interval_collisions, last_collision_time, collision_cooldown):
    current_time = time.time()
    pair_key = tuple(sorted((car1.id, car2.id)))

    if car1.spawned_recently() or car2.spawned_recently():
        return  # Ignore collisions for newly spawned cars

    if current_time - last_collision_time[pair_key] > collision_cooldown:
        last_collision_time[pair_key] = current_time
        if pair_key not in collisions:
            collisions[pair_key] = True
            interval_collisions.append(pair_key)

def count_collisions(collisions):
    total = 0
    for cars, col in list(collisions.items()):
        if col:
            total += 1
    return total

def can_create(car_list, path):
    if len(car_list) >= MAX_TOTAL_CARS:
        return False

    cars_in_same_direction = sum(1 for car in car_list if car.path == path)
    if cars_in_same_direction >= MAX_CARS_PER_DIRECTION:
        return False

    for car in car_list:
        if car.path == path:
            if path == Paths.TOP_BOTTOM:
                if is_close_to(car.x_pos, car.y_pos, (SCREEN_WIDTH / 2) - 8, RENDER_BORDER, 10):
                    return False
            elif path == Paths.BOTTOM_TOP:
                if is_close_to(car.x_pos, car.y_pos, (SCREEN_WIDTH / 2) + 8, SCREEN_HEIGHT - RENDER_BORDER, 10):
                    return False
            elif path == Paths.LEFT_RIGHT:
                if is_close_to(car.x_pos, car.y_pos, RENDER_BORDER, (SCREEN_HEIGHT / 2) + 8, 10):
                    return False
            elif path == Paths.RIGHT_LEFT:
                if is_close_to(car.x_pos, car.y_pos, SCREEN_WIDTH - RENDER_BORDER, (SCREEN_HEIGHT / 2) - 8, 10):
                    return False
    return True

def create_cars(i, bottom_top_next_interval, left_right_next_interval, cars, config): # TODO: Are the cars actually created with each interval? The exact count sometimes gets skipped.
    if i % bottom_top_next_interval == 0:
        if can_create(cars, Paths.BOTTOM_TOP):
            cars.append(return_car(Paths.BOTTOM_TOP, config))

    if i % left_right_next_interval == 0:
        if can_create(cars, Paths.LEFT_RIGHT):
            cars.append(return_car(Paths.LEFT_RIGHT, config))

# --- PLOTTING --- TODO: Actually display plots

def initialize_plot():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    ax1.set_xlabel('Interval')
    ax1.set_ylabel('Count')
    ax1.set_title('Collisions Over Time')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Interval')
    ax2.set_ylabel('Count')
    ax2.set_title('Crossings Over Time')
    ax2.legend()
    ax2.grid(True)

    ax3.set_xlabel('Interval')
    ax3.set_ylabel('Rewards')
    ax3.set_title('Rewards Over Time')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    return fig, ax1, ax2, ax3

def update_plot(collision_records, intersection_records, reward_records, fig, ax1, ax2, ax3):
    ax1.plot(range(1, len(collision_records) + 1), collision_records, label='Collisions', marker='o')
    ax2.plot(range(1, len(intersection_records) + 1), intersection_records, label='Crossings', marker='x')
    ax3.plot(range(1, len(reward_records) + 1), reward_records, label='Rewards', marker='s')

    for ax in (ax1, ax2, ax3):
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()


def save_and_plot_data(interval_count, collision_records, intersection_records, reward_records, parameter_records):
    # Ensure all inputs are lists
    if not isinstance(interval_count, list):
        interval_count = [interval_count]
    if not isinstance(collision_records, list):
        collision_records = [collision_records]
    if not isinstance(intersection_records, list):
        intersection_records = [intersection_records]
    if not isinstance(reward_records, list):
        reward_records = [reward_records]

    # Create DataFrame
    data = pd.DataFrame({
        'Interval': interval_count,
        'Collisions': collision_records,
        'Crossings': intersection_records,
        'Rewards': reward_records
    })

    # Append the data to the CSV file without overwriting
    data.to_csv('simulation_data.csv', mode='a', header=not pd.io.common.file_exists('simulation_data.csv'), index=False)

    # Create a DataFrame for the current interval's parameters
    parameters_data = pd.DataFrame(parameter_records, columns=['Interval', 'Max_Velocity', 'Acceleration', 'Collision_Distance'])

    # Append the parameters data to the CSV file without overwriting
    parameters_data.to_csv('parameter_data.csv', mode='a', header=not pd.io.common.file_exists('parameter_data.csv'), index=False)

# --- AGENT AND INTERSECTION FLOW ---
def update_parameters(config, action):
    max_velocity, acceleration, collision_distance = action
    config.update_parameters(
        max_velocity=max_velocity,
        acceleration=acceleration,
        collision_distance=collision_distance,
    )

def initialize_records(collision_records, intersection_records, reward_records, parameter_records):
    if collision_records is None:
        collision_records = []
    if intersection_records is None:
        intersection_records = []
    if reward_records is None:
        reward_records = []
    if parameter_records is None:
        parameter_records = []
    return collision_records, intersection_records, reward_records, parameter_records

def initialize_state_and_action(agent, bottom_top_next_interval, left_right_next_interval, is_first_interval, last_collisions):
    state = np.array([
        bottom_top_next_interval,
        left_right_next_interval,
        1 if is_first_interval else 0,
        last_collisions
    ])
    action = agent.select_action(state)
    return state, action

# --- MAIN SIMULATION FUNCTION ---

def run_simulation(config, agent, interval_count=0, collision_records=None, intersection_records=None, reward_records=None, parameter_records=None):

    collision_records, intersection_records, reward_records, parameter_records = initialize_records(
        collision_records, intersection_records, reward_records, parameter_records)

    screen, clock = initialize_window()

    running = True
    start_time = pygame.time.get_ticks()

    cars = []
    bottom_top_interval = 75
    left_right_interval = 75
    bottom_top_next_interval = bottom_top_interval + randint(-10, 10)
    left_right_next_interval = left_right_interval + randint(-10, 10)

    interval_start_time = start_time
    start_collisions = 0
    interval_crossings = 0

    interval_results = []
    last_collision_time = defaultdict(lambda: 0)
    collision_cooldown = 0

    collisions = {}
    interval_collisions = []
    total_crossings = 0
    crossed_cars = set()

    total_reward = 0
    is_first_interval = True
    i = 0  # Initialize the loop counter

    state, action = initialize_state_and_action(agent, bottom_top_next_interval, left_right_next_interval,
                                                1 if is_first_interval else 0, 0)

    update_parameters(config, action)
    parameter_records.append([interval_count, *action])  # Log initial parameters

    while running:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - start_time
        interval_elapsed_time = current_time - interval_start_time

        if elapsed_time > 40005:
            running = False

        end_collisions = count_collisions(collisions)
        interval_collisions_count = len(interval_collisions)

        reward = interval_crossings - interval_collisions_count * 100
        reward = max(min(reward, 200), -500)
        total_reward += reward

        collision_records.append(interval_collisions_count)
        intersection_records.append(interval_crossings)
        reward_records.append(reward)

        last_collisions = interval_collisions_count
        last_crossings = interval_crossings

        if interval_elapsed_time > 10000: # TODO: Adjust sequence and make code cleaner
            bottom_top_next_interval = bottom_top_interval + randint(-10, 10)
            left_right_next_interval = left_right_interval + randint(-10, 10)

            state = np.array([
                bottom_top_next_interval,
                left_right_next_interval,
                1 if is_first_interval else 0,
                last_collisions,
            ])

            action = agent.select_action(state)
            update_parameters(config, action)
            parameter_records.append([interval_count, *action])
            interval_count += 1
            is_first_interval = False
            interval_results.append((interval_collisions_count, interval_crossings, is_first_interval, bottom_top_next_interval, left_right_next_interval, reward))
            interval_start_time = current_time
            save_and_plot_data(interval_count, interval_collisions_count, interval_crossings, reward, parameter_records)
            interval_crossings = 0
            interval_elapsed_time = 0
            interval_collisions.clear()

        create_cars(i, bottom_top_next_interval, left_right_next_interval, cars, config)

        handle_events(cars)

        update_screen(screen, total_crossings, collisions, interval_count, config)

        for car in cars[:]:
            if car.at_border():
                if car.crossed_intersection and car.id not in crossed_cars:
                    total_crossings += 1
                    interval_crossings += 1
                    crossed_cars.add(car.id)
                car.remove_from_simulation()
                cars.remove(car)
            else:
                car.draw(screen)
                car.update(cars, i, speed_factor=SPEED_FACTOR)
                if car.crossed_intersection and car.id not in crossed_cars:
                    total_crossings += 1
                    interval_crossings += 1
                    crossed_cars.add(car.id)

        for j, car1 in enumerate(cars):
            for car2 in cars[j+1:]:
                if is_close_to(car1.x_pos, car1.y_pos, car2.x_pos, car2.y_pos, 10.5):
                    add_collision(car1, car2, collisions, interval_collisions, last_collision_time, collision_cooldown)

        pygame.display.flip()
        clock.tick(144)
        i += 1

    pygame.quit()
    return interval_results, bottom_top_next_interval, left_right_next_interval, total_reward, collision_records, intersection_records, reward_records, interval_count, parameter_records

if __name__ == "__main__":
    print("Please run `train.py` to test the simulation")