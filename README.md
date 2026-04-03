# OfflineTaxiRL
TaxiRL implementation using Offline RL

## Heuristic Policy Scoring

The log-generation policy chooses which passenger to target with this score:

`score = payout - ALPHA_PICKUP * pickup_distance - BETA_TRIP * trip_distance`

### What each term means

- `payout`: the reward for successfully delivering that passenger.
- `pickup_distance`: the Manhattan distance from the taxi's current position to the passenger's pickup location.
- `trip_distance`: the Manhattan distance from that passenger's pickup location to their dropoff location.

### Why `ALPHA_PICKUP` and `BETA_TRIP` exist

- `ALPHA_PICKUP` controls how much the policy penalizes going far just to reach a passenger.
- `BETA_TRIP` controls how much the policy penalizes choosing a passenger whose ride will be long after pickup.

These are hand-tuned weights. They let the heuristic balance three competing goals:

- prefer passengers with higher payout
- prefer passengers that are closer to reach
- prefer passengers whose full trip is not too expensive in steps

In this project:

- `ALPHA_PICKUP = 0.8`
- `BETA_TRIP = 0.35`

That means pickup distance is penalized more heavily than trip distance. The heuristic is therefore more sensitive to how far the taxi must travel before service can even begin.

### How the score is used

The score is recomputed at every step while the taxi is not carrying a passenger.

- The heuristic evaluates every waiting passenger.
- It picks the passenger with the highest score.
- It moves one step toward that passenger's pickup point.

Once a passenger has been picked up, the policy stops rescoring candidates and commits to that passenger's dropoff location until the ride is completed.

## Reward Design

The project uses a simple step-based reward design:

- valid move: `-0.1`
- invalid move: `-1.0`
- successful dropoff: add the passenger payout

### Why this reward exists

- The small negative reward for a valid move encourages shorter, more efficient routes.
- The larger negative reward for an invalid move discourages wasteful actions such as pushing into grid boundaries.
- The positive dropoff reward makes successful task completion the main objective.

### Reward formula per step

For a step with no successful dropoff:

- valid move -> `-0.1`
- invalid move -> `-1.0`

For a step with a successful dropoff:

- reward = move penalty + passenger payout

Example:

- if the taxi makes a valid move and drops off a passenger with payout `12`
- step reward = `-0.1 + 12 = 11.9`
