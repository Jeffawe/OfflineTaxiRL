import random
from typing import List

from environment.gridWorld import GridWorld
from environment.passenger import Passenger
from environment.taxi import Taxi

class TaxiManager:
    def __init__(self, width, height, num_passengers):
        self.width = width
        self.height = height
        self.num_passengers = num_passengers
        self.taxi: Taxi = Taxi(0, 0)
        self.grid: GridWorld = GridWorld(width, height)
        self.locations: set[tuple[int, int]] = set()
        self.passengers: List[Passenger] = list()
        self.completed_passengers: set[Passenger] = set()

    def create_passengers(self):
        for i in range(0, self.num_passengers):
            random_pickup_location = self.get_unique_location()
            random_destination_location = self.get_unique_location()
            payout = random.randint(3, 20)
            self.passengers.append(Passenger(random_pickup_location[0], random_pickup_location[1], random_destination_location[0], random_destination_location[1], payout))

    def get_unique_location(self) -> tuple[int, int]:
        while True:
            random_x = random.randint(1, self.width - 1)
            random_y = random.randint(1, self.height - 1)

            if (random_x, random_y) not in self.locations:
                self.locations.add((random_x, random_y))
                return random_x, random_y

    def pickup_passenger(self):
        for passenger in self.passengers:
            if passenger.pickup_position() == self.taxi.position():
                self.taxi.pickup(passenger)
                return True

        return False

    def dropoff_passenger(self):
        current_passenger = self.taxi.current_passenger
        dropped_off = self.taxi.can_dropoff()
        if dropped_off and current_passenger is not None:
            self.completed_passengers.add(current_passenger)
            return True

        return False

    def is_done(self) -> bool:
        return len(self.completed_passengers) == self.num_passengers

    def move_taxi(self, dx: int, dy: int) -> bool:
        next_x = self.taxi.x + dx
        next_y = self.taxi.y + dy

        if not self.grid.is_in_bounds(next_x, next_y):
            return False

        self.taxi.move(dx, dy)
        return True

    def render(self) -> str:
        passenger_positions = {
            passenger.pickup_position()
            for passenger in self.passengers
            if not passenger.picked_up and not passenger.completed
        }
        destination_positions = {
            passenger.dropoff_position()
            for passenger in self.passengers
            if passenger.picked_up and not passenger.completed
        }

        rows = []
        for y in range(self.height):
            cells = []
            for x in range(self.width):
                position = (x, y)
                if position == self.taxi.position():
                    cells.append("T")
                elif position in passenger_positions:
                    cells.append("P")
                elif position in destination_positions:
                    cells.append("D")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))

        return "\n".join(rows)


