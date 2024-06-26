import pygame
from car import Car
from config import *
from random import randint

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Car Simulation')
    clock = pygame.time.Clock()
    running = True
    
    cars = []

    i = 0
    while running:
        # Add cars conditionally
        if (i % (144*2) == 30 and len(cars) < 4):
            cars.append(Car(StartingPos.TOP, Paths.TOP_BOTTOM, randint(0, 100)))
            cars.append(Car(StartingPos.BOTTOM, Paths.BOTTOM_TOP, randint(0, 100)))
        if (i % (144*2) == 0 and len(cars) < 4):
            cars.append(Car(StartingPos.LEFT, Paths.LEFT_RIGHT, randint(0, 100)))
        if (i % (144*2) == 0 and len(cars) < 4):
            cars.append(Car(StartingPos.RIGHT, Paths.RIGHT_LEFT, randint(0, 100)))

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
        for car in cars[:]:
            if car.at_border():
                car.remove_from_simulation()
                cars.remove(car)
            else:
                car.draw(screen)
                car.update(cars, i)

        pygame.display.flip()
        clock.tick(144)
        i += 1

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
