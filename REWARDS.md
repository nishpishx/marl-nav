```python
from reward import RacecarReward

reward_fn = RacecarReward()
reward = reward_fn.reward(agent_id="A", state=state, action=action)
```

`state["A"]` is expected to look like:

```python
{
    "wall_collision": False,
    "opponent_collisions": [],
    "pose": [...],       # x, y, z, roll, pitch, yaw
    "velocity": [...],   # x, y, z, roll, pitch, yaw
    "progress": 0.23,    # progress within the current lap, [0, 1]
    "lap": 1,
    "time": 12.4,
    "checkpoint": 3,
    "rank": 2,
    "wrong_way": False,
    "observations": {...},
}
```

The action dict is the actual racecar action space for that scenario, usually:

```python
{"motor": 0.2, "steering": -0.1}
```

or:

```python
{"speed": 0.2, "steering": -0.1}
```

Reward shape:

- Progress delta: positive when the car moves forward around the track
- Wall/opponent collisions: large negative penalties
- Wrong way: extra negative penalty
- Action thrash: small penalty when controls change too abruptly
- Forward motion: small bonus when the velocity points along the car heading
