import matplotlib.pyplot as plt
from matplotlib import ticker

# Fill in your value: count
counts_010 = {
    10: 10,
    11: 10,
    3: 13,
    8: 10,
    9: 9,
    12: 18,
}

counts = counts_010

# Sort by value (x-axis)
x = sorted(counts)
y = [counts[val] for val in x]

# Plot
plt.bar(x, y)
plt.xlabel("Episode")
plt.ylabel("Count")
plt.title("Early stopping at episode")
plt.xticks(range(max(x) + 1))
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)
plt.savefig("early-stopping-at-episode.png")
plt.show()