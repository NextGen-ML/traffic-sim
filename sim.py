import pygame
from car import Car
from config import *
from random import randint
import sys

total_crossings = 0
start_crossings = 0
crossed_cars = set()

def is_close_to(x1, y1, x2, y2, tolerance):
    dist = abs((((x2-x1)**2) + ((y2-y1)**2))**0.5)
    return tolerance > dist

def return_car(path, config):
    if path == Paths.TOP_BOTTOM:
        return Car(StartingPos.TOP, Paths.TOP_BOTTOM, randint(0, 100000), config)
    elif path == Paths.BOTTOM_TOP:
        return Car(StartingPos.BOTTOM, Paths.BOTTOM_TOP, randint(0,100000), config)
    elif path == Paths.LEFT_RIGHT:
        return Car(StartingPos.LEFT, Paths.LEFT_RIGHT, randint(0,100000), config)
    elif path == Paths.RIGHT_LEFT:
        return Car(StartingPos.RIGHT, Paths.RIGHT_LEFT, randint(0, 100000), config)
    
    return None

collisions = {}

def add_collision(car1, car2):
    if (car1.id, car2.id) not in collisions:
        print("💥COLLISION")
        collisions[(car1.id, car2.id)] = True

def count_collisions():
    global collisions
    total = 0  # Initialize total to 0
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
                    if is_close_to(car.x_pos, car.y_pos, (SCREEN_WIDTH / 2)-8, RENDER_BORDER, 10):
                        can = False
                elif path == Paths.BOTTOM_TOP:
                    if is_close_to(car.x_pos, car.y_pos, (SCREEN_WIDTH / 2)+8, SCREEN_HEIGHT-RENDER_BORDER, 10):
                        can = False
                elif path == Paths.LEFT_RIGHT:
                    if is_close_to(car.x_pos, car.y_pos, RENDER_BORDER, (SCREEN_HEIGHT / 2)+ 8, 10):
                        can = False
                elif path == Paths.RIGHT_LEFT:
                    if is_close_to(car.x_pos, car.y_pos, SCREEN_WIDTH-RENDER_BORDER, (SCREEN_HEIGHT / 2)- 8, 2):
                        can = False
    return can

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Car Simulation')
    clock = pygame.time.Clock()
    running = True
    start_time = pygame.time.get_ticks()
    start_crossings = 0

    config = Config()  # Create the config instance
    cars = []
    bottom_top_interval = 50  # Base interval in frames for generating BOTTOM_TOP cars
    left_right_interval = 50  # Base interval in frames for generating LEFT_RIGHT cars

    # Generate random initial intervals within +- 15 of the base interval
    bottom_top_next_interval = bottom_top_interval + randint(-10, 10)
    left_right_next_interval = left_right_interval + randint(-10, 10)

    collision_records = []  # List to record the number of collisions at each interval
    intersection_records = []  # List to record the number of crossings at each interval
    interval_start_time = start_time
    start_collisions = count_collisions()  # Record collisions at the start of the interval
    total_crossings = 0  # Total number of cars crossing the intersection

    i = 0
    while running:

        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - start_time
        interval_elapsed_time = current_time - interval_start_time

        if elapsed_time > 60001:  
            running = False

        if interval_elapsed_time >= 20000: 
            end_collisions = count_collisions()
            interval_collisions = end_collisions - start_collisions  
            collision_records.append(interval_collisions) 
            start_collisions = end_collisions  
            interval_start_time = current_time  
            bottom_top_next_interval = bottom_top_interval + randint(-10, 10)  
            left_right_next_interval = left_right_interval + randint(-10, 10) 
            interval_crossings = total_crossings - start_crossings
            intersection_records.append(interval_crossings)
            start_crossings = total_crossings

            # Calculate and record the number of cars that crossed the intersection in this interval
            # interval_crossings = count_crossed_intersections(cars) - total_crossings
            # intersection_records.append(interval_crossings)
            # total_crossings += interval_crossings
            for car in cars[:]:
                if car.at_border():
                    if car.crossed_intersection and car.id not in crossed_cars:
                        total_crossings += 1
                        crossed_cars.add(car.id)
                    car.remove_from_simulation()
                    cars.remove(car)
                else:
                    car.draw(screen)
                    car.update(cars, i, speed_factor=SPEED_FACTOR)
                    if car.crossed_intersection and car.id not in crossed_cars:
                        total_crossings += 1
                        crossed_cars.add(car.id)

        # Generate BOTTOM_TOP cars at regular intervals
        if i % bottom_top_next_interval == 0:
            if can_create(cars, Paths.BOTTOM_TOP):
                cars.append(return_car(Paths.BOTTOM_TOP, config))

        # Generate LEFT_RIGHT cars at regular intervals
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

        # Screen clearing and rectangle drawing
        screen.fill((255, 255, 255))
        rect_width, rect_height = 100, 100
        rect_x = (SCREEN_WIDTH / 2) - (rect_width / 2)
        rect_y = (SCREEN_HEIGHT / 2) - (rect_height / 2)
        pygame.draw.rect(screen, (0, 0, 0), (rect_x, rect_y, rect_width, rect_height), width=5)

        font = pygame.font.Font(None, 16)  
        collision_font = pygame.font.Font(None, 32) 

        # Display the configuration parameters on the left side
        params_text = [
            f"MAX_VELOCITY: {config.MAX_VELOCITY}",
            f"ACCELERATION: {config.ACCELERATION}",
            f"COLLISION_DISTANCE: {config.COLLISION_DISTANCE}",
            f"WAIT_TIME: {config.WAIT_TIME}",
            f"DISTANCE_BETWEEN_CARS: {config.DISTANCE_BETWEEN_CARS}"
        ]

        for idx, text in enumerate(params_text):
            param_surface = font.render(text, True, (0, 0, 0))
            screen.blit(param_surface, (10, 10 + idx * 20))  

        collisions_text = collision_font.render(f"Collisions: {count_collisions()}", True, (255, 0, 0))
        screen.blit(collisions_text, (350, 10)) 

        crossings_text = collision_font.render(f"Crossings: {total_crossings}", True, (0, 0, 255))
        screen.blit(crossings_text, (350, 50))

        # Update cars
        for car in cars[:]:
            if car.at_border():
                if car.crossed_intersection and car.id not in crossed_cars:
                    total_crossings += 1
                    crossed_cars.add(car.id)
                car.remove_from_simulation()
                cars.remove(car)
            else:
                car.draw(screen)
                car.update(cars, i, speed_factor=SPEED_FACTOR)
                if car.crossed_intersection and car.id not in crossed_cars:
                    total_crossings += 1
                    crossed_cars.add(car.id)
                
        prev_car = None

        # Check for collisions
        for j, car1 in enumerate(cars):
            for car2 in cars[j+1:]:
                if is_close_to(car1.x_pos, car1.y_pos, car2.x_pos, car2.y_pos, 15):
                    add_collision(car1, car2)

            prev_car = car

        pygame.display.flip()
        clock.tick(144)
        i += 1

    pygame.quit()

    for idx, (collisions, crossings) in enumerate(zip(collision_records, intersection_records)):
        print(f"Interval {idx + 1}: Collisions: {collisions}, Crossings: {crossings}")

    return total_crossings

if __name__ == "__main__":
    total_crossings = run_simulation()
    print(f"Total Collisions: {count_collisions()}")
    print(f"Total Crossings: {total_crossings}")
    sys.exit()