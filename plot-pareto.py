"""Plot pareto front from evolution output json"""
# Usage:
# - arg1 - path to the file with all solutions
# - arg2 - path to the file with front
# - arg3 - plot title

import json
import sys

import matplotlib.pyplot as plt

from main import ACCURACY_THRESHOLD

def plot_pareto_front(solutions_all, solutions_front, title):
    with open(solutions_all, "r") as f:
        solutions = json.load(f)
    with open(solutions_front, "r") as f:
        front = json.load(f)

    plt.figure(figsize=(8, 6))

    sizes = [sol["size"] for sol in solutions]
    accuracies = [sol["accuracy"] if sol["accuracy"] >= ACCURACY_THRESHOLD else sol["accuracy"] * 10 for sol in solutions]
    plt.scatter(sizes, accuracies, c="blue", label=f"All Solutions ({len(sizes)})")

    sizes = [sol["size"] for sol in front]
    accuracies = [sol["accuracy"] if sol["accuracy"] >= ACCURACY_THRESHOLD else sol["accuracy"] * 10 for sol in front]
    plt.scatter(sizes, accuracies, c="red", label=f"Pareto Front ({len(sizes)})")

    plt.xlabel("Model Size")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xlim(2000, 4800)
    # plt.ylim(0.845, 0.92)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(solutions_all[:-5] + ".png")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: script.py <path_to_all_solutions> <path_to_front_solutions>")
        sys.exit(1)
    plot_pareto_front(sys.argv[1], sys.argv[2], sys.argv[3])
