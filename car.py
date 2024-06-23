from config import *

class Car:
    def __init__(self, starting_pos, path, id):
        self.starting_pos = starting_pos
        self.x_pos = SCREEN_WIDTH / 2 
        self.y_pos = SCREEN_HEIGHT / 2

        if starting_pos == StartingPos.BOTTOM:
            self.y_pos = SCREEN_HEIGHT - RENDER_BORDER
            self.x_pos+=8
        elif starting_pos == StartingPos.TOP:
            self.y_pos = RENDER_BORDER
            self.x_pos-=8
        elif starting_pos == StartingPos.LEFT:
            self.x_pos = RENDER_BORDER
            self.y_pos += 8
        elif starting_pos == StartingPos.RIGHT:
            self.x_pos = SCREEN_WIDTH - RENDER_BORDER
            self.y_pos -= 8
        
        self.id = id
        self.path = path
        self.vel = MAX_VELOCITY
        self.accel = 0
        self.row = None 
        self.starting = False
        self.queue:CarQueue = None
        
        self.vx, self.vy, self.ax, self.ay = 0, 0, 0, 0 # bad syntax but too lazy to fix
        self.nearby_cars = []

    def set_row(self, row):
        self.row = row
        
    def get_row(self):
        return self.row
    
    # Useless for now, could be useful later
    def output(self):
        return {"x_pos":self.x_pos, "y_pos":self.y_pos, "vx":self.vx, "vy":self.vy, "ax":self.ax, "ay":self.ay, "path":self.path, "queue":self.queue, "row":self.row}
    
    def get_cars(self, all_cars):
        self.nearby_cars = sorted(
            (car for car in all_cars if self.distance_to(car) < 200 and car.id != self.id),
            key=lambda car: self.distance_to(car)
        )
        
    def distance_to_intersection(self):
        intersection_x = SCREEN_WIDTH / 2
        intersection_y = SCREEN_HEIGHT / 2
        
        distance = ((self.x_pos - intersection_x) ** 2 + (self.y_pos - intersection_y) ** 2) ** 0.5
        return distance
    
    def will_collide(self, other_car, time_steps=100, time_interval=0.1):

        for step in range(time_steps):
            time = (step + 1) * time_interval
            my_future_x = self.x_pos + self.vx * time
            my_future_y = self.y_pos + self.vy * time
            other_future_x = other_car.x_pos + other_car.vx * time
            other_future_y = other_car.y_pos + other_car.vy * time
            
            distance = ((my_future_x - other_future_x) ** 2 + (my_future_y - other_future_y) ** 2) ** 0.5
            
            if distance < COLLISION_DISTANCE:
                return True

        return False

        
    def distance_to(self, other):
        other: Car
        return ((self.x_pos - other.x_pos) ** 2 + (self.y_pos - other.y_pos) ** 2) ** 0.5
    
    def calculate_row(self):
        car: Car

        for car in self.nearby_cars:
            if car.queue is None:
                
                if car.path == self.path: # check for equiv. paths as well
                    print("hello")
                    if car.row == None or car.row == True:
                        self.row == True
                else:
                    if self.will_collide(car):
                        print("COLLISION")
                        if self.distance_to_intersection() < car.distance_to_intersection():
                            
                            self.row = True
                            
                        elif self.distance_to_intersection() == car.distance_to_intersection():
                            print("mismatch")
                            
                            if self.path == Paths.BOTTOM_TOP or self.path == Paths.TOP_BOTTOM:
                                self.row = True
                            else:
                                self.row = False
                        else:
                            self.row = False
        #             else:
        #                 self.row = True
    

    def update(self, cars):
        self.get_cars(cars)
        
        if len(self.nearby_cars) > 0:
            self.calculate_row()
        
        if not self.at_border():
            dt = 1/FRAME_RATE
            if self.row == None:
                self.vx, self.vy = self.move_in_direction()
            elif self.row is False: # Add other paths
                if self.path == Paths.BOTTOM_TOP:
                    if self.ay == 0:
                        deltaY = ((SCREEN_HEIGHT / 2) + 50) - self.y_pos
                        self.ay = (0-(self.vy**2))/(2*deltaY)
                        print(self.ay)
                elif self.path == Paths.TOP_BOTTOM:
                    if self.ay == 0:
                        deltaY = ((SCREEN_HEIGHT / 2) - 50) - self.y_pos
                        self.ay = (0-(self.vy**2))/(2*deltaY)
                        print(self.ay)
                
                elif self.path == Paths.LEFT_RIGHT:
                    if self.ax == 0:
                        print(f"initial: {self.ax}")
                        deltaX = ((SCREEN_WIDTH / 2) - 50) - self.x_pos
                        self.ax = (0-(self.vx**2))/(2*deltaX)
                        print(deltaX)
                        
                elif self.path == Paths.RIGHT_LEFT:
                    if self.ax == 0:
                        print(f"initial: {self.ax}")
                        deltaX = ((SCREEN_WIDTH / 2) + 50) - self.x_pos
                        self.ax = (0-(self.vx**2))/(2*deltaX)
                        print(self.ax)
            
            elif self.row is True:
                if self.vx == 0 or self.vy == 0:
                    self.starting = True
                    
                if not self.starting:
                    self.vx, self.vy = self.move_in_direction()
                else: # Add other paths
                    if self.path is Paths.BOTTOM_TOP:
                        self.ay = -ACCELERATION
                        self.starting = True
                    elif self.path is Paths.TOP_BOTTOM:
                        self.ay = ACCELERATION
                        self.starting = True
                        
                    if self.path is Paths.LEFT_RIGHT:
                        self.ax = ACCELERATION
                        self.starting = True
                    elif self.path is Paths.RIGHT_LEFT:
                        self.ax = ACCELERATION
                        self.starting = True
            
            if abs(self.vx) < 0.1 and not self.starting:
                self.vx = 0
                self.ax = 0
                
            if abs(self.vy) < 0.1 and not self.starting:
                self.vy = 0
                self.ay = 0 
                
            if self.starting_pos == StartingPos.BOTTOM:
                self.vy = min(self.vy, 0)  # Prevent moving up
            elif self.starting_pos == StartingPos.TOP:
                self.vy = max(self.vy, 0)  # Prevent moving down
            elif self.starting_pos == StartingPos.LEFT:
                self.vx = max(self.vx, 0)  # Prevent moving right
            elif self.starting_pos == StartingPos.RIGHT:
                self.vx = min(self.vx, 0)  # Prevent moving left
            
            # clamp velocities
            self.vx = max(min(self.vx, MAX_VELOCITY), -MAX_VELOCITY)
            self.vy = max(min(self.vy, MAX_VELOCITY), -MAX_VELOCITY)
            
            if (abs(self.vx) == MAX_VELOCITY or abs(self.vy) == MAX_VELOCITY) and self.starting:
                self.starting = False

            # Euler method for integration
            self.vx += self.ax*dt
            self.vy += self.ay*dt
            
            self.x_pos += (self.vx * dt)
            self.y_pos += (self.vy * dt)
            
    # helper function to make stupid code easier
    def move_in_direction(self):
        if self.starting_pos == StartingPos.BOTTOM:
            return (0, -self.vel)
        elif self.starting_pos == StartingPos.TOP:
            return (0, self.vel)
        elif self.starting_pos == StartingPos.LEFT:
            return (self.vel, 0)
        elif self.starting_pos == StartingPos.RIGHT:
            return (-self.vel, 0)

    def at_border(self):
        if self.starting_pos == StartingPos.BOTTOM:
            return self.y_pos <= RENDER_BORDER
        elif self.starting_pos == StartingPos.TOP:
            return self.y_pos >= SCREEN_HEIGHT - RENDER_BORDER
        elif self.starting_pos == StartingPos.LEFT:
            return self.x_pos >= SCREEN_WIDTH - RENDER_BORDER
        elif self.starting_pos == StartingPos.RIGHT:
            return self.x_pos <= RENDER_BORDER
        return False
    
    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), (self.x_pos, self.y_pos, 10, 10))

