from fairproj.bases import Allocation, Agent, E
import random as ran 
from typing import List, Callable, Tuple


def get_random_weights(n: int = 5) -> List[float]: 
    lst = [ran.random() for _ in range(n)]
    s = sum(lst)
    return [i/s for i in lst]


def create_utilfunc(weights: List[float]) -> Callable: 
    # --- sanity check --- 
    if abs(sum(weights) - 1) > E: 
        raise ValueError("We assume the utility function is normalised. Sum of all weights must be one.") 
    
    piece_size = 1 / len(weights)
    def utilfunc(left: float, right: float) -> float: 
        total = 0.
        for i, w in enumerate(weights): 
            piece_start = i * piece_size
            piece_end = (i + 1) * piece_size
            overlap = max(0, min(right, piece_end) - max(left, piece_start))
            total += w * (overlap / piece_size)
        return total
    
    return utilfunc


def envy_report(alloc: Allocation) -> str: 
    report = ""
    for agent in alloc.assignments: 
        report +=  f"{agent}\n"
        for other, bundle in alloc.assignments.items(): 
            value = sum(agent.eval(p) for p in bundle)
            report += f"  valuation for {other.name}'s bundle: {value:.3f}\n"
    
    return report
            
            
def boxed(lines: list[str], title: str = "") -> str:
    width = max(len(l) for l in lines)
    if title:
        width = max(width, len(title) + 2)
    
    top    = f"┌─ {title} {'─' * (width - len(title) - 1)}┐" if title else f"┌{'─' * (width + 2)}┐"
    bottom = f"└{'─' * (width + 2)}┘"
    middle = [f"│ {l:<{width}} │" for l in lines]
    
    return "\n".join([top, *middle, bottom])


def make_up_agents(n: int) -> Tuple[List, List[Agent]]: 
    weights = [get_random_weights() for _ in range(n)]
    agents = [Agent(f"Voter {i+1}", create_utilfunc(weights[i])) 
              for i in range(n)]
    return weights, agents 
