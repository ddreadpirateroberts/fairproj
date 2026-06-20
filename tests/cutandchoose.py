from fairproj.protocols import CutAndChoose
from fairproj.bases import Agent
from fairproj.utils import (
    get_random_weights, create_utilfunc, envy_report, boxed
)


if __name__ == "__main__":
    weights = [get_random_weights(), get_random_weights()]
    agents = [
        Agent("Voter 1", create_utilfunc(weights[0])),
        Agent("Voter 2", create_utilfunc(weights[1]))
    ]

    alloc = CutAndChoose(agents).run()

    print(boxed([
        f"Voter 1 weights: {[round(w, 3) for w in weights[0]]}",
        f"Voter 2 weights: {[round(w, 3) for w in weights[1]]}",
    ], title="Preferences"))

    print(boxed(str(alloc).splitlines(), title="Allocation"))

    print(boxed([
        f"Proportional : {alloc.is_proportional()}",
        f"Envy-free    : {alloc.is_envy_free()}",
    ], title="Fairness"))

    print(boxed(envy_report(alloc).splitlines(), title="Envy Report"))
