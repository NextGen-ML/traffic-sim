from config import *

class Car:
    def __init__(self, starting_pos, path):
        self.starting_pos = starting_pos
        self.x_pos = SCREEN_WIDTH / 2 
        self.y_pos = SCREEN_HEIGHT / 2

        # Adjust position based on starting location
        if starting_pos == StartingPos.BOTTOM:
            self.y_pos = SCREEN_HEIGHT - RENDER_BORDER
        elif starting_pos == StartingPos.TOP:
            self.y_pos = RENDER_BORDER
        elif starting_pos == StartingPos.LEFT:
            self.x_pos = RENDER_BORDER
        elif starting_pos == StartingPos.RIGHT:
            self.x_pos = SCREEN_WIDTH - RENDER_BORDER
        
        self.path = path
        self.vel = MAX_VELOCITY
        self.accel = 0
        self.row = None 
        self.starting = False
        
        self.vx, self.vy, self.ax, self.ay = 0, 0, 0, 0 # bad syntax but too lazy to fix

    def set_row(self, row):
        self.row = row
        
    def get_row(self):
        return self.row

    def update(self):
        if not self.at_border():
            dt = 1/FRAME_RATE
            if self.row == None:
                self.vx, self.vy = self.move_in_direction()
            elif self.row is False: # Add other paths
                if self.path == Paths.BOTTOM_TOP:
                    if self.ay == 0:
                        deltaY = (SCREEN_HEIGHT / 2) - self.y_pos
                        self.ay = (0-(self.vy**2))/(2*deltaY)
                        print(self.ay)
            
            elif self.row is True:
                if self.vx == 0 or self.vy == 0:
                    self.starting = True
                    
                if not self.starting:
                    self.vx, self.vy = self.move_in_direction()
                else: # Add other paths
                    if self.path is Paths.BOTTOM_TOP:
                        self.ay = -ACCELERATION
                        self.starting = True
            
            if abs(self.vx) < 0.1 and not self.starting:
                self.vx = 0
                self.ax = 0
                
            if abs(self.vy) < 0.1 and not self.starting:
                self.vy = 0
                self.ax = 0 
            
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
        pygame.draw.rect(screen, (255, 0, 0), (self.x_pos, self.y_pos, 20, 10))

