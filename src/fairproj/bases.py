from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Union, overload
from dataclasses import dataclass, field

E = 1e-4

@dataclass
class Piece:
    """A contiguous interval [left, right] of the cake."""
    left: float
    right: float


class Agent:
    def __init__(self, name: str, util_func: Callable[[float, float], float]):
        self.name = name
        self._util = util_func
        
    def __repr__(self):
            return f"Agent({self.name!r})"
        
    @overload
    def eval(self, piece: Piece) -> float: ...
    @overload
    def eval(self, left: float, right: float) -> float: ...

    def eval( # type: ignore[misc]
        self, 
        left_or_piece: Union[Piece, float], 
        right: Optional[float] = None
    ) -> float:
        if isinstance(left_or_piece, Piece):
            return self._util(left_or_piece.left, left_or_piece.right)
        if right is None: 
            raise ValueError("right must be provided when left is a float")
        return self._util(left_or_piece, right)

    def mark(self, target_value: float, start: float = 0, 
             end: float = 1, tol=1e-6) -> float:
        """
        Returns the cut point `end` such that eval(start, end) == target_value.
        Binary search over the cake.
        """
        if abs(self.eval(start, end) - target_value) < 1e-9: 
            return end 
        
        lo, hi = start, end
        while hi - lo > tol:
            mid = (lo + hi) / 2
            mid_val = self.eval(start, mid)
            if abs(mid_val - target_value) < tol:
                return mid
            elif mid_val < target_value:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    
    def pick_from(self, pieces: List[Piece]) -> Tuple[List[Piece], List]: 
        """
        Pick the best piece for agent across the pieces. 
        Returns the pick, and the remaining pieces.
        """
        best_pick = max(pieces, key=lambda p: self.eval(p))
        return [best_pick], [p for p in pieces if p != best_pick]
        

@dataclass
class Allocation:
    """Maps each agent name to the list of pieces they received."""
    assignments: dict[Agent, list[Piece]] = field(default_factory=dict)

    def __str__(self): 
        report = []
        for agent, bundle in self.assignments.items(): 
            total = self.value_for(agent)
            bundle_str = ", ".join(f"[{p.left:.3f}, {p.right:.3f}]" for p in bundle)
            report.append(f"{agent.name}: {bundle_str} (value: {total})")
        return "\n".join(report)

    @overload
    def assign(self, agent: Agent, piece: Piece) -> None: ...
    @overload
    def assign(self, assignments: dict[Agent, list[Piece]]) -> None: ...

    def assign( # type: ignore[misc]
        self, 
        agent_or_dict: Union[Agent, dict[Agent, list[Piece]]], 
        piece: Optional[Piece] = None
    ) -> None:
        if isinstance(agent_or_dict, dict):
            for agent, pieces in agent_or_dict.items():
                for piece in pieces:
                    self.assignments.setdefault(agent, []).append(piece)
        else:
            assert piece is not None, "piece must be provided when agent is passed directly"
            self.assignments.setdefault(agent_or_dict, []).append(piece)

    def value_for(self, agent: Agent) -> float:
        pieces = self.assignments.get(agent, [])
        return sum(agent.eval(p.left, p.right) for p in pieces)

    # ---- fairness auditing ----
    
    def is_proportional(self) -> bool:
        n = len(self.assignments)
        return all(
            self.value_for(a) >= 1 / n - E
            for a in self.assignments
        )

    def is_envy_free(self) -> bool:
        for agent in self.assignments:
            my_value = self.value_for(agent)
            for other, pieces in self.assignments.items():
                if other != agent:
                    other_value = sum(agent.eval(p) for p in pieces)
                    if other_value > my_value + E:
                        return False
        return True


class FairDivisionProtocol(ABC):
    """Base class for all fair division protocols."""
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self._validate()

    def _validate(self):
        """Override to add agent-count or other precondition checks."""
        pass

    @abstractmethod
    def run(self) -> Allocation:
        """Execute the protocol and return an allocation."""
        ...


class ProportionalProtocol(FairDivisionProtocol):
    """Guarantees each agent at least 1/n of their value."""
    pass


class EnvyFreeProtocol(FairDivisionProtocol):
    """Guarantees no agent prefers another's allocation. Implies proportionality."""
    pass


class TwoAgentProtocol(FairDivisionProtocol):
    def _validate(self):
        if len(self.agents) != 2:
            raise ValueError(f"{type(self).__name__} requires exactly 2 agents.")


class ThreeAgentProtocol(FairDivisionProtocol):
    def _validate(self):
        if len(self.agents) != 3:
            raise ValueError(f"{type(self).__name__} requires exactly 3 agents.")


class StromquistConvergenceError(Exception):
    def __init__(self, delta: float):
        super().__init__(f"Stromquist failed to converge with delta={delta}. ")
