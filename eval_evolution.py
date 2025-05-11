import json
from types import SimpleNamespace

from deap import creator, base
from deap.base import Fitness
from deap.tools import sortNondominated
from matplotlib import pyplot as plt
from pymoo.indicators.hv import HV
import numpy as np
from pymoo.indicators.igd import IGD

REF_POINT = np.array([4801, 0.1])
CUTOFF_ACCURACY =  0.87
EXPERIMENTS = {
    "Experiment #010": "../experiments/010/pareto_solutions_gen_14.json",
    "Experiment #011": "../experiments/011/pareto_solutions_gen_15.json",
    "Experiment #016": "../experiments/016/pareto_solutions_gen_16.json",
    "Random Search #017": "../experiments/017/pareto_solutions.json",
}

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # minimize size, maximize accuracy
creator.create("Individual", dict, fitness=creator.FitnessMulti)
all_sols = []


class MyFitness(Fitness):
    weights = (4800.0, -1.0)


i = 0
fitness_objs = []
for key, value in EXPERIMENTS.items():
    with open(value, "r") as f:
        sols = json.load(f)
        for s in sols:
            if s["accuracy"] < CUTOFF_ACCURACY:
                continue
            fit = MyFitness()
            fit.values = (s["accuracy"], s["size"])
            ind = SimpleNamespace(fitness=fit, idx=i)
            i+=1
            fitness_objs.append(ind)
            s["experiment"] = key
            all_sols.append(s)
        sols = np.array([(s["fitness"][0], -s["fitness"][1]) for s in sols if s["accuracy"] >= CUTOFF_ACCURACY])

    ind = HV(ref_point=REF_POINT)
    print(f"Hypervolume for {key}: {ind(sols)}")

# IGD
front_indices = sortNondominated(fitness_objs, k=len(fitness_objs), first_front_only=True)[0]
pareto_solutions = [all_sols[i.idx] for i in front_indices]
ind = IGD(np.array([(p["fitness"][0], p["fitness"][1]) for p in pareto_solutions]))
for key, value in EXPERIMENTS.items():
    with open(value, "r") as f:
        sols = json.load(f)
        sols = np.array([(s["fitness"][0], s["fitness"][1]) for s in sols if s["accuracy"] >= CUTOFF_ACCURACY])
        print(f"IDG for {key}: {ind(sols)}")


# Plot all solutions

# plt.figure(figsize=(12, 9))  # Width: 12 inches, Height: 8 inches

COLORS = ["blue", "red", "green", "orange"]
for i, (key, value) in enumerate(EXPERIMENTS.items()):
    with open(value, "r") as f:
        sols = json.load(f)
        sols = np.array([(s["fitness"][0], s["fitness"][1]) for s in sols if s["accuracy"] >= CUTOFF_ACCURACY])
        sizes = [s[0] for s in sols]
        accuracies = [s[1] for s in sols]
        plt.scatter(sizes, accuracies,edgecolors=COLORS[i], alpha=0.8, facecolors="none")
    solutions = [s for s in pareto_solutions if s["experiment"] == key]
    sizes = [s["size"] for s in solutions]
    accuracies = [s["accuracy"] for s in solutions]
    plt.scatter(sizes, accuracies, c=COLORS[i], label=f"{key} ({len(solutions)})")
plt.xlabel('Size')
plt.ylabel('Accuracy')
plt.title('Experiment contributions to final Pareto front')
plt.legend()
plt.grid(True)
plt.savefig("comparison.png", dpi=600, bbox_inches='tight')
plt.show()

RANDOM_EXPERIMENTS = {
    "Random Search #015": "../experiments/015/pareto-solutions.json"
}
for key, value in RANDOM_EXPERIMENTS.items():
    with open(value, "r") as f:
        sols = json.load(f)
        sols = np.array([(s["size"], -s["accuracy"]) for s in sols if s["accuracy"] >= CUTOFF_ACCURACY])


    ind = HV(ref_point=REF_POINT)
    print(f"Hypervolume for {key}: {ind(sols)}")
