class Passenger:
    def __init__(self, pickup_x, pickup_y, dropoff_x, dropoff_y, payout):
        self.name = None
        self.pickup_x = pickup_x
        self.pickup_y = pickup_y
        self.dropoff_x = dropoff_x
        self.dropoff_y = dropoff_y
        self.payout = payout

        self.picked_up = False
        self.completed = False

    def pickup_position(self):
        return (self.pickup_x, self.pickup_y)

    def dropoff_position(self):
        return (self.dropoff_x, self.dropoff_y)
