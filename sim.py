import pygame
from car import Car
from config import *
from random import randint

def is_close_to(x1, y1, x2, y2, tolerance):
    dist = abs((((x2-x1)**2) + ((y2-y1)**2))**0.5)
    return tolerance > dist

def return_car(path):
    if path == Paths.TOP_BOTTOM:
        return Car(StartingPos.TOP, Paths.TOP_BOTTOM, randint(0, 100))
    elif path == Paths.BOTTOM_TOP:
        return Car(StartingPos.BOTTOM, Paths.BOTTOM_TOP, randint(0,100))
    elif path == Paths.LEFT_RIGHT:
        return Car(StartingPos.LEFT, Paths.LEFT_RIGHT, randint(0,100))
    elif path == Paths.RIGHT_LEFT:
        return Car(StartingPos.RIGHT, Paths.RIGHT_LEFT, randint(0, 100))
    
    return None

collisions = {}

def add_collision(car1, car2):
    if (car1.id, car2.id) not in collisions:
        collisions[(car1.id, car2.id)] = True
        

def count_collisions():
    global collisions
    total = 0
    for cars, col in list(collisions.items()):
        if col:
            total +=1
    
    return total

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
    
    cars = []

    i = 0
    while running:
        
        current_time = pygame.time.get_ticks()
        if (current_time - start_time) > 30000:  # Stop after 30000 milliseconds
            running = False

        if (i % (144*2) == 30):
            if can_create(cars, Paths.TOP_BOTTOM):
                cars.append(return_car(Paths.TOP_BOTTOM))
            if can_create(cars, Paths.BOTTOM_TOP):
                cars.append(return_car(Paths.BOTTOM_TOP))
        if (i % (144*2) == 0):
            if can_create(cars, Paths.LEFT_RIGHT):
                cars.append(return_car(Paths.LEFT_RIGHT))
        if (i % (144*2) == 0):
            if can_create(cars, Paths.RIGHT_LEFT):
                cars.append(return_car(Paths.RIGHT_LEFT))

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

        # Update cars
        prev_car = None
        for car in cars[:]:
            if car.at_border():
                car.remove_from_simulation()
                cars.remove(car)
            else:
                car.draw(screen)
                car.update(cars, i)
            if prev_car is not None:
                if is_close_to(car.x_pos, car.y_pos, prev_car.x_pos, prev_car.y_pos, 10):
                    add_collision(car, prev_car)
            
            prev_car = car

        pygame.display.flip()
        clock.tick(144)
        i += 1

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
    print(f"Collisions: {count_collisions()}")
    
