rewards = reward_fn(
    prev_state=prev_state,
    actions=actions,
    next_state=next_state,
    events=events,
)

rewards returns:
{
    "taxi_0": -1.0,
    "taxi_1": 20.0,
    "taxi_2": -6.0,
}

Every timestep:              -1
Successful dropoff:          +100
Collision/blocking:          -5 to -20
Move closer to target:       small positive shaping
Move away from target:       small negative shaping