from car import Car
from config import *

import pygame

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Car Simulation')
    clock = pygame.time.Clock()
    running = True
    
    cars = []
    
    cars.append(Car(StartingPos.TOP, Paths.TOP_BOTTOM, 1))
    cars.append(Car(StartingPos.LEFT, Paths.LEFT_RIGHT, 2))

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_k:
                    temp = not cars[0].get_row() if cars[0].get_row() is not None else False
                    cars[0].set_row(temp)

        screen.fill((255, 255, 255))
        
        # line_x_position = SCREEN_WIDTH / 2 + 50
        # pygame.draw.line(screen, (0, 0, 255), (line_x_position, 0), (line_x_position, SCREEN_HEIGHT), 5)

        car: Car
        
        cars = [car for car in cars if not car.at_border()]
        for car in cars:
            print(car.get_row())
            car.draw(screen)
            car.update(cars)
                


        pygame.display.flip()
        clock.tick(144)

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
