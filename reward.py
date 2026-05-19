"""Reward logic for destination-only taxi navigation.

How this module works:
- Each taxi only needs two pieces of state: `pos` and `destination`.
- Every step starts with a small step cost.
- When a taxi reaches its destination, it gets an arrival bonus.
- A portion of that bonus can be shared with the rest of the team.
- Illegal actions, collisions, blocked moves, and idle actions each add a penalty.
- A small shaping reward uses grid distance to the destination, so moving
  closer helps and moving away hurts.
"""

from typing import Any, Callable, Dict, Optional


AgentID = str
State = Dict[str, Any]
ActionDict = Dict[AgentID, int]
EventDict = Dict[AgentID, Dict[str, Any]]
RewardDict = Dict[AgentID, float]


class RewardConfig:
    def __init__(
        self,
        step_cost: float = -1.0,
        arrival_bonus: float = 100.0,
        illegal_action: float = -10.0,
        collision: float = -10.0,
        blocked: float = -2.0,
        idle: float = -0.5,
        distance_shaping: float = 1.0,
        gamma: float = 0.95,
        team_share: float = 0.25,
    ):
        self.step_cost = step_cost
        self.arrival_bonus = arrival_bonus
        self.illegal_action = illegal_action
        self.collision = collision
        self.blocked = blocked
        self.idle = idle
        self.distance_shaping = distance_shaping
        self.gamma = gamma
        self.team_share = team_share


def grid_distance(a, b) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class DestinationReward:
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        distance_fn: Callable = grid_distance,
    ):
        self.config = config if config is not None else RewardConfig()
        self.distance_fn = distance_fn

    def __call__(
        self,
        prev_state: State,
        actions: ActionDict,
        next_state: State,
        events: EventDict,
    ) -> RewardDict:
        """
        Expected inputs:

        actions:
            {
                "taxi_0": 0,
                "taxi_1": 3,
            }

        events:
            {
                "taxi_0": {
                    "illegal_action": False,
                    "collision": False,
                    "blocked": False,
                    "idle": False,
                }
            }

        state:
            {
                "taxis": {
                    "taxi_0": {
                        "pos": (row, col),
                        "destination": (row, col),
                    }
                }
            }

        Each taxi only needs its current position and its destination.
        """

        cfg = self.config
        agents = list(actions.keys())
        rewards = {agent: cfg.step_cost for agent in agents}

        def add(agent: AgentID, amount: float):
            rewards[agent] += amount

        def add_arrival_bonus(actor: AgentID, amount: float):
            """
            Splits the arrival bonus between the taxi that reached the goal
            and the rest of the team.

            Example with team_share = 0.25 and arrival_bonus = 100:
                actor gets 75 directly
                all agents split 25
            """
            individual_amount = amount * (1.0 - cfg.team_share)
            shared_amount = amount * cfg.team_share

            rewards[actor] += individual_amount

            if agents:
                per_agent = shared_amount / len(agents)
                for agent in agents:
                    rewards[agent] += per_agent

        for agent in agents:
            ev = events.get(agent, {})
            destination = next_state["taxis"][agent].get("destination")
            if destination is None:
                destination = prev_state["taxis"][agent].get("destination")

            if self._reached_destination(prev_state, next_state, agent, destination):
                add_arrival_bonus(agent, cfg.arrival_bonus)

            if ev.get("illegal_action", False):
                add(agent, cfg.illegal_action)

            if ev.get("collision", False):
                add(agent, cfg.collision)

            if ev.get("blocked", False):
                add(agent, cfg.blocked)

            if ev.get("idle", False):
                add(agent, cfg.idle)

            shaping = self._potential_based_shaping(prev_state, next_state, agent, destination)
            add(agent, shaping)

        return rewards

    def _potential_based_shaping(
        self,
        prev_state: State,
        next_state: State,
        agent: AgentID,
        destination,
    ) -> float:
        """
        Potential-based shaping:
            F(s, s') = gamma * Phi(s') - Phi(s)

        Here:
            Phi(s) = -distance_to_destination

        Moving closer gives a positive reward.
        Moving farther gives a negative reward.
        """

        cfg = self.config

        if destination is None:
            return 0.0

        prev_pos = prev_state["taxis"][agent]["pos"]
        next_pos = next_state["taxis"][agent]["pos"]

        prev_phi = -self.distance_fn(prev_pos, destination)
        next_phi = -self.distance_fn(next_pos, destination)

        return cfg.distance_shaping * ((cfg.gamma * next_phi) - prev_phi)

    def _reached_destination(
        self,
        prev_state: State,
        next_state: State,
        agent: AgentID,
        destination,
    ) -> bool:
        prev_pos = prev_state["taxis"][agent]["pos"]
        next_pos = next_state["taxis"][agent]["pos"]

        return prev_pos != destination and next_pos == destination


MultiTaxiReward = DestinationReward
