# car_queue.py
import threading
import time
from config import *

class CarQueue:
    def __init__(self, host_car, config):
        self.host_car = host_car
        self.config = config
        self.motion_path_queue = {host_car.path: [host_car]}
        self.lock = threading.Lock()
        self.active = True
        self.queue_start_time = {host_car.path: time.time()}  # Track when the queue started waiting
        self.thread = threading.Thread(target=self.run_queue_management)
        self.thread.start()

    def join(self, car):
        with self.lock:
            car_exists = any(car.id == existing_car.id for path in self.motion_path_queue.values() for existing_car in path)
            if not car_exists:
                if car.path in self.motion_path_queue:
                    self.motion_path_queue[car.path].append(car)
                elif get_partner_path(car.path) in self.motion_path_queue:
                    self.motion_path_queue[get_partner_path(car.path)].append(car)
                else:
                    self.motion_path_queue[car.path] = [car]
                    self.queue_start_time[car.path] = time.time()  # Start the timer when a car joins an empty queue
                car.queue = self

    def update_queue(self, i):
        with self.lock:
            for path, cars in list(self.motion_path_queue.items()):
                self.motion_path_queue[path] = [car for car in cars if not car.has_crossed_intersection() or not car.at_border()]

                if len(self.motion_path_queue[path]) == 0:
                    del self.motion_path_queue[path]
                    del self.queue_start_time[path]  # Reset the timer when the queue becomes empty

            self.check_host_car()

    def check_host_car(self):
        if self.motion_path_queue:
            top_path = next(iter(self.motion_path_queue))
            self.host_car = self.motion_path_queue[top_path][0]

    def run_queue_management(self):
        while self.active:
            time.sleep(float(self.config.WAIT_TIME))
            self.reorder_queue()

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
                for car in cars:
                    car.row = False

    def shutdown(self):
        with self.lock:
            self.active = False
            if self.thread.is_alive():
                self.thread.join()

    def get_wait_times(self):
        with self.lock:
            current_time = time.time()
            wait_times = {path: current_time - self.queue_start_time[path] for path in self.queue_start_time}
        return wait_times

    def reset_wait_times(self):
        with self.lock:
            self.queue_start_time = {path: time.time() for path in self.queue_start_time}