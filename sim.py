import pygame
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from car import Car
from config import *
from random import randint
import sys
import numpy as np
import threading
from matplotlib.animation import FuncAnimation
from intersection import Intersection

total_crossings = 0
crossed_cars = set()

collisions = {}
collision_records = []  
intersection_records = []  
reward_records = []  
interval_count = 0


# Define intersection
four_way = Intersection(
    motion_path_array=[Paths.TOP_BOTTOM, Paths.BOTTOM_TOP, Paths.LEFT_RIGHT, Paths.RIGHT_LEFT],
    number_of_roads=4,
    starting_positions=[StartingPos.BOTTOM, StartingPos.TOP, StartingPos.LEFT, StartingPos.RIGHT],
    size=(100, 100)
)

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

def add_collision(car1, car2):
    if (car1.id, car2.id) not in collisions:
        collisions[(car1.id, car2.id)] = True

def count_collisions():
    global collisions
    total = 0
    for cars, col in list(collisions.items()):
        if col:
            total += 1
    return total

def count_crossed_intersections(cars):
    return sum(1 for car in cars if car.crossed_intersection)

def can_create(car_list, path):
    can = True
    if car_list != []:
        for car in car_list:
            if car.path == path:
                if path == Paths.TOP_BOTTOM:
                    if is_close_to(car.x_pos, car.y_pos, (SCREEN_WIDTH / 2) - 8, RENDER_BORDER, 10):
                        can = False
                elif path == Paths.BOTTOM_TOP:
                    if is_close_to(car.x_pos, car.y_pos, (SCREEN_WIDTH / 2) + 8, SCREEN_HEIGHT - RENDER_BORDER, 10):
                        can = False
                elif path == Paths.LEFT_RIGHT:
                    if is_close_to(car.x_pos, car.y_pos, RENDER_BORDER, (SCREEN_HEIGHT / 2) + 8, 10):
                        can = False
                elif path == Paths.RIGHT_LEFT:
                    if is_close_to(car.x_pos, car.y_pos, SCREEN_WIDTH - RENDER_BORDER, (SCREEN_HEIGHT / 2) - 8, 2):
                        can = False
    return can

plt.ion() 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
collision_line, = ax1.plot([], [], label='Collisions', marker='o')
crossing_line, = ax2.plot([], [], label='Crossings', marker='x')
reward_line, = ax3.plot([], [], label='Rewards', marker='s')

def initialize_plot():
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

def update_plot(collision_records, intersection_records, reward_records):
    collision_line.set_data(range(1, len(collision_records) + 1), collision_records)
    crossing_line.set_data(range(1, len(intersection_records) + 1), intersection_records)
    reward_line.set_data(range(1, len(reward_records) + 1), reward_records)
    
    for ax in (ax1, ax2, ax3):
        ax.relim()
        ax.autoscale_view()
    
    fig.canvas.draw()
    fig.canvas.flush_events()

def save_and_plot_data():
    # Save to CSV
    data = pd.DataFrame({
        'Interval': range(1, len(collision_records) + 1),
        'Collisions': collision_records,
        'Crossings': intersection_records,
        'Rewards': reward_records
    })
    data.to_csv('simulation_data.csv', index=False)

    # Update the plot
    update_plot(collision_records, intersection_records, reward_records)

def update_parameters(config, action):
    max_velocity, acceleration, collision_distance, wait_time, distance_between_cars = action
    config.update_parameters(
        max_velocity=max_velocity,
        acceleration=acceleration,
        collision_distance=collision_distance,
        wait_time=wait_time,
        distance_between_cars=distance_between_cars
    )

def run_simulation(config, agent):
    global total_crossings, collision_records, intersection_records, reward_records, interval_count

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Car Simulation')
    clock = pygame.time.Clock()
    running = True
    start_time = pygame.time.get_ticks()

    initialize_plot()

    cars = []
    bottom_top_interval = 50
    left_right_interval = 50
    bottom_top_next_interval = bottom_top_interval + randint(-10, 10)
    left_right_next_interval = left_right_interval + randint(-10, 10)

    interval_start_time = start_time
    start_collisions = count_collisions()
    interval_crossings = 0

    i = 0
    interval_results = []
    is_first_interval = True
    last_parameter_update = start_time

    while running:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - start_time
        interval_elapsed_time = current_time - interval_start_time

        if elapsed_time > 80003:  
            running = False

        if interval_elapsed_time >= 20000:
            interval_count += 1
            end_collisions = count_collisions()
            interval_collisions = end_collisions - start_collisions
            interval_crossings = interval_crossings

            interval_results.append((interval_collisions, interval_crossings, is_first_interval, bottom_top_next_interval, left_right_next_interval))
            
            collision_records.append(interval_collisions)
            intersection_records.append(interval_crossings)
            
            reward = interval_crossings - interval_collisions * 500
            reward_records.append(reward)
            
            start_collisions = end_collisions
            interval_crossings = 0
            interval_start_time = current_time
            is_first_interval = False  # Set to False after the first interval
            state = np.array([
                # collision_records[-1] if collision_records else 0,
                # intersection_records[-1] if intersection_records else 0,
                bottom_top_next_interval,
                left_right_next_interval,
                # len(four_way.motion_path_array),
                # four_way.number_of_roads,
                # four_way.size[0],
                # four_way.size[1],
                1 if is_first_interval else 0,
            ])
            action = agent.select_action(state)
            update_parameters(config, action)

        # Generate cars
        if i % bottom_top_next_interval == 0:
            if can_create(cars, Paths.BOTTOM_TOP):
                cars.append(return_car(Paths.BOTTOM_TOP, config))

        if i % left_right_next_interval == 0:
            if can_create(cars, Paths.LEFT_RIGHT):
                cars.append(return_car(Paths.LEFT_RIGHT, config))

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_k and cars:
                    temp = not cars[0].get_row() if cars[0].get_row() is not None else False
                    cars[0].set_row(temp)

        # Screen clearing and drawing
        screen.fill((255, 255, 255))
        rect_width, rect_height = 100, 100
        rect_x = (SCREEN_WIDTH / 2) - (rect_width / 2)
        rect_y = (SCREEN_HEIGHT / 2) - (rect_height / 2)
        pygame.draw.rect(screen, (0, 0, 0), (rect_x, rect_y, rect_width, rect_height), width=5)

        # Display parameters and stats
        font = pygame.font.Font(None, 16)  
        collision_font = pygame.font.Font(None, 26) 
        interval_font = pygame.font.Font(None, 26)

        params = config.get_parameters()
        params_text = [
            f"MAX_VELOCITY: {params['MAX_VELOCITY']}",
            f"ACCELERATION: {params['ACCELERATION']}",
            f"COLLISION_DISTANCE: {params['COLLISION_DISTANCE']}",
            f"WAIT_TIME: {params['WAIT_TIME']}",
            f"DISTANCE_BETWEEN_CARS: {params['DISTANCE_BETWEEN_CARS']}"
        ]

        for idx, text in enumerate(params_text):
            param_surface = font.render(text, True, (0, 0, 0))
            screen.blit(param_surface, (10, 10 + idx * 20))  

        collisions_text = collision_font.render(f"Collisions: {count_collisions()}", True, (255, 0, 0))
        screen.blit(collisions_text, (340, 10)) 

        crossings_text = collision_font.render(f"Crossings: {total_crossings}", True, (0, 0, 255))
        screen.blit(crossings_text, (340, 35))

        interval_text = interval_font.render(f"{interval_count}", True, (80, 80, 80))
        screen.blit(interval_text, (20, SCREEN_HEIGHT - 50))

        # Update and draw cars
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

        # Check for collisions
        for j, car1 in enumerate(cars):
            for car2 in cars[j+1:]:
                if is_close_to(car1.x_pos, car1.y_pos, car2.x_pos, car2.y_pos, 15):
                    add_collision(car1, car2)

        pygame.display.flip()
        clock.tick(144)
        i += 1

    pygame.quit()

    for idx, (collisions, crossings, first_interval, bottom_top_next_interval, left_right_next_interval) in enumerate(interval_results):
        print(f"Interval {idx + 1}: Collisions: {collisions}, Crossings: {crossings}, First Interval: {first_interval}, Bottom-Top Interval: {bottom_top_next_interval}, Left-Right Interval: {left_right_next_interval}")

    save_and_plot_data()

    return interval_results, bottom_top_next_interval, left_right_next_interval

if __name__ == "__main__":
    from policy_agent import PolicyGradientAgent
    from environment import IntersectionEnv, four_way
    from config import Config
    
    config = Config()
    env = IntersectionEnv(config, four_way, None)
    agent = PolicyGradientAgent(env)
    env.agent = agent

    total_crossings = run_simulation(config, agent)
    print(f"Total Collisions: {count_collisions()}")
    print(f"Total Crossings: {total_crossings}")
    
    plt.ioff() 
    plt.show()