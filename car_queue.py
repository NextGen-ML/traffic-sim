import threading

class CarQueue:
    def __init__(self, host_car):
        self.host_car = host_car
        self.motion_path_queue = {host_car.path: [host_car]}
        self.lock = threading.Lock()
        self.timer = None
        self.active = True

    def join(self, car):
        with self.lock:
            # Search for any car with the same ID in all paths before adding
            car_exists = any(car.id == existing_car.id for path in self.motion_path_queue.values() for existing_car in path)

            if not car_exists:
                if car.path in self.motion_path_queue:
                    self.motion_path_queue[car.path].append(car)
                else:
                    self.motion_path_queue[car.path] = [car]
                car.queue = self


    def update_queue(self, i):
        with self.lock:
            current_top_path = next(iter(self.motion_path_queue)) if self.motion_path_queue else None

            for path, cars in list(self.motion_path_queue.items()):
                self.motion_path_queue[path] = [car for car in cars if not car.has_crossed_intersection() or car.at_border()]
                

                if len(self.motion_path_queue[path]) == 1:
                    if path == current_top_path:
                        first_path, first_cars = next(iter(self.motion_path_queue.items()))
                        del self.motion_path_queue[first_path]
                        self.motion_path_queue[first_path] = first_cars
                        self.update_host_car()
                
                if len(self.motion_path_queue[path])==0:
                    del self.motion_path_queue[path]
                
        
                    

                        
            
            if (i % 144 == 0):
                print(self.motion_path_queue)

            if self.active:
                if self.timer:
                    self.timer.cancel()
                self.timer = threading.Timer(10, self.reorder_queue)
                self.timer.start()


    def reorder_queue(self):
        with self.lock:
            if self.motion_path_queue:
                first_path, first_cars = next(iter(self.motion_path_queue.items()))
                del self.motion_path_queue[first_path]
                self.motion_path_queue[first_path] = first_cars
                self.update_host_car()

    def update_host_car(self):
        if self.motion_path_queue:
            first_path = next(iter(self.motion_path_queue))
            self.host_car = self.motion_path_queue[first_path][0]
            for path, cars in self.motion_path_queue.items():
                if path == first_path:
                    for car in cars:
                        car.row = True
                else:
                    for car in cars:
                        car.row = False

    def shutdown(self):
        with self.lock:
            self.active = False
            if self.timer:
                self.timer.cancel()
