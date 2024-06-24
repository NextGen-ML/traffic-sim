from car import Car
from config import *
from random import randint

import pygame

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Car Simulation')
    clock = pygame.time.Clock()
    running = True
    
    cars = []
    
    cars.append(Car(StartingPos.LEFT, Paths.LEFT_RIGHT, 2))
    
    i = 0

    while running:
        
        if ( i % (144*2) == 30 and len(cars) < 3):
            cars.append(Car(StartingPos.TOP, Paths.TOP_BOTTOM, randint(0, 100)))
        
        if (i >(144*2)):
            i = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_k:
                    temp = not cars[0].get_row() if cars[0].get_row() is not None else False
                    cars[0].set_row(temp)

        screen.fill((255, 255, 255))
        
        rect_width = 100  # 50 pixels from the center on each side
        rect_height = 100  # 50 pixels from the center on each side
        rect_x = (SCREEN_WIDTH / 2) - (rect_width / 2)  # Center the rectangle
        rect_y = (SCREEN_HEIGHT / 2) - (rect_height / 2)  # Center the rectangle

        # Draw the rectangle
        pygame.draw.rect(screen, (0, 0,  255), (rect_x, rect_y, rect_width, rect_height))
        

        car: Car
        
        cars = [car for car in cars if not car.at_border()]
        for car in cars:
            # print(car.get_row())
            car.draw(screen)
            car.update(cars)
                


        pygame.display.flip()
        clock.tick(144)
        i+=1

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
