from fairproj.protocols import EvenPaz
from fairproj.utils import (
    make_up_agents, envy_report, boxed
)

if __name__ == "__main__":
    n = int(input("How many agents? "))
    
    weights, agents = make_up_agents(n)

    model = EvenPaz(agents)
    
    alloc = model.run()
    
    print(boxed([
        f"Voter {i+1} weights: {[round(w, 3) for w in weights[i]]}"
        for i in range(n)
    ]))
    
    print(boxed(str(alloc).splitlines(), title="Allocation"))

    print(boxed([
        f"Proportional : {alloc.is_proportional()}",
        f"Envy-free    : {alloc.is_envy_free()}",
    ], title="Fairness"))

    print(boxed(envy_report(alloc).splitlines(), title="Envy Report"))
