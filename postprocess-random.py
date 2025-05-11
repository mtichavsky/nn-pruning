import json
import sys
from types import SimpleNamespace

from deap.base import Fitness
from deap.tools import sortNondominated

class MyFitness(Fitness):
    weights = (4800.0, -1.0)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: postprocess-random.py <path_to_all_solutions>")
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        solutions = json.load(f)

    fitness_objs = []
    for i, s in enumerate(solutions):
        fit = MyFitness()
        fit.values = (s["accuracy"], s["size"])
        ind = SimpleNamespace(fitness=fit, idx=i)
        fitness_objs.append(ind)

    objs = [(s["accuracy"], -s["size"]) for s in solutions]
    front_indices = sortNondominated(fitness_objs, k=len(fitness_objs), first_front_only=True)[0]
    pareto_solutions = [solutions[i.idx] for i in front_indices]

    with open("pareto_solutions.json", "w") as f:
        json.dump(pareto_solutions, f)