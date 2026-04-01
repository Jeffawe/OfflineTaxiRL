from environment.taxiManager import TaxiManager

def main() -> None:
    manager = TaxiManager(width=5, height=5, num_passengers=2)
    manager.create_passengers()

    controls = {
        "w": (0, -1),
        "a": (-1, 0),
        "s": (0, 1),
        "d": (1, 0),
    }

    print("Taxi environment")
    print("Controls: w=up, a=left, s=down, d=right, q=exit")

    while True:
        print()
        print(manager.render())
        command = input("Move: ").strip().lower()

        if command == "q":
            print("Exiting.")
            break

        if command not in controls:
            print("Invalid input. Use w, a, s, d, or q.")
            continue

        dx, dy = controls[command]
        if not manager.move_taxi(dx, dy):
            print("Out of bounds.")
            continue

        manager.pickup_passenger()
        manager.dropoff_passenger()

        if manager.is_done():
            print()
            print(manager.render())
            print("All passengers completed. Exiting.")
            break


if __name__ == "__main__":
    main()
