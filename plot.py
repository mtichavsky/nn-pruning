"""Plot pareto front from evolution output json"""
# Usage:
# - arg1 - path to the file with all solutions
# - arg2 - path to the file with front
# - arg3 - plot title

import json
import sys

import matplotlib.pyplot as plt


def plot_pareto_front(solutions_all, solutions_front, title):
    with open(solutions_all, "r") as f:
        solutions = json.load(f)
    with open(solutions_front, "r") as f:
        front = json.load(f)

    plt.figure(figsize=(8, 6))

    sizes = [sol["size"] for sol in solutions]
    # accuracies = [sol["accuracy"] for sol in solutions]
    accuracies = [sol["accuracy"] if sol["accuracy"] >= 0.8 else sol["accuracy"] * 10 for sol in solutions]
    print("Number of points: ", len(sizes))
    plt.scatter(sizes, accuracies, c="blue", label="All Solutions")
    sizes = [sol["size"] for sol in front]
    # accuracies = [sol["accuracy"] for sol in front ]
    accuracies = [sol["accuracy"] if sol["accuracy"] >= 0.8 else sol["accuracy"] * 10 for sol in front ]
    print("Number of points: ", len(sizes))
    plt.scatter(sizes, accuracies, c="red", label="Pareto Front")

    plt.xlabel("Model Size")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xlim(0, 4800)
    # plt.ylim(0.80, 1.0)
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(solutions_all[:-5] + ".png")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: script.py <path_to_all_solutions> <path_to_front_solutions>")
        sys.exit(1)
    plot_pareto_front(sys.argv[1], sys.argv[2], sys.argv[3])
