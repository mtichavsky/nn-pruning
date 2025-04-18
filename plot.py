"""Plot pareto front from evolution output json"""

import json
import sys

import matplotlib.pyplot as plt


def plot_pareto_front(json_path):
    with open(json_path, "r") as f:
        solutions = json.load(f)

    sizes = [sol["size"] for sol in solutions]
    accuracies = [sol["accuracy"] for sol in solutions]
    print("Number of points: ", len(sizes))

    plt.figure(figsize=(8, 6))

    plt.scatter(sizes, accuracies, c="blue", label="Pareto Front")
    plt.xlabel("Model Size")
    plt.ylabel("Accuracy")
    plt.title("Pareto Front")
    plt.xlim(4800, 0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(sys.argv[1][:-5] + ".png")


if __name__ == "__main__":
    plot_pareto_front(sys.argv[1])
