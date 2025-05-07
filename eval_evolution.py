import json

from pymoo.indicators.hv import HV
import numpy as np

REF_POINT = np.array([4801, 0.1])
CUTOFF_ACCURACY =  0.85
EXPERIMENTS = {
    "010": "../experiments/010/pareto_solutions_gen_14.json",
    "011": "../experiments/011/pareto_solutions_gen_15.json",
}
for key, value in EXPERIMENTS.items():
    with open(value, "r") as f:
        sols = json.load(f)
        sols = np.array([(s["fitness"][0], -s["fitness"][1]) for s in sols if s["accuracy"] >= CUTOFF_ACCURACY])


    ind = HV(ref_point=REF_POINT)
    print(f"Hypervolume for experiment #{key}: {ind(sols)}")

RANDOM_EXPERIMENTS = {
    "015": "../experiments/015/pareto-solutions.json"
}
for key, value in RANDOM_EXPERIMENTS.items():
    with open(value, "r") as f:
        sols = json.load(f)
        sols = np.array([(s["size"], -s["accuracy"]) for s in sols if s["accuracy"] >= CUTOFF_ACCURACY])


    ind = HV(ref_point=REF_POINT)
    print(f"Hypervolume for experiment #{key}: {ind(sols)}")
