from typing import Optional

from environment.passenger import Passenger

class Taxi:
    def __init__(self, x, y):
        self.current_passenger: Optional[Passenger] = None
        self.x = x
        self.y = y

    def get_current_passenger(self):
        return self.current_passenger

    def has_passenger(self):
        return self.current_passenger is not None

    def pickup(self, passenger):
        if self.current_passenger is None and passenger.picked_up == False:
            self.current_passenger = passenger
            passenger.picked_up = True

    def dropoff(self):
        if self.current_passenger is not None:
            self.current_passenger.completed = True
            self.current_passenger = None

    def can_dropoff(self):
        if self.current_passenger is not None:
            if self.x == self.current_passenger.dropoff_x and self.y == self.current_passenger.dropoff_y:
                self.dropoff()
                return True

        return False

    def position(self):
        return (self.x, self.y)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
