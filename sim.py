from car import Car
from config import *

import pygame

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Car Simulation')
    clock = pygame.time.Clock()
    running = True

    car = Car(StartingPos.BOTTOM, Paths.BOTTOM_TOP)
    car_visible = True 

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_k:
                    if car.get_row() == None:  
                        car.set_row(False)
                    else:
                        var = car.get_row()
                        car.set_row(not var)
                        print(car.get_row())

        screen.fill((255, 255, 255))

        if car_visible:
            car.update()
            if car.at_border():
                car_visible = False
            else:
                car.draw(screen)

        pygame.display.flip()
        clock.tick(144)

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
