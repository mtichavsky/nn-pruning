import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Layer sizes as provided
# TODO check this
layer_sizes = {
    'conv1': 64,
    'layer1': 256,
    'layer2': 640,
    'layer3': 1280,
    'layer4': 2560
}

# Read the JSON file
with open('comparison_front.json', 'r') as f:
    data = json.load(f)

# Initialize dictionary to store masks for each layer
layer_masks = {
    'conv1': [],
    'layer1': [],
    'layer2': [],
    'layer3': [],
    'layer4': []
}

# Process each entry's mask
for entry in data:
    mask = entry['mask']
    current_idx = 0
    
    # Split mask into layers based on sizes
    for layer_name, size in layer_sizes.items():
        layer_mask = mask[current_idx:(current_idx + size)]
        pruned_percentage = (1 - sum(layer_mask) / len(layer_mask)) * 100
        layer_masks[layer_name].append(pruned_percentage)
        current_idx += size

# Print average pruning percentages
print("Average pruning percentages per layer:")
for layer_name, percentages in layer_masks.items():
    avg_pruning = np.mean(percentages)
    print(f"{layer_name}: {avg_pruning:.2f}%")

# Create box plot
plt.figure(figsize=(10, 6))
# sns.violinplot(data=list(layer_masks.values()), orient='v')
plt.boxplot(layer_masks.values(), tick_labels=layer_masks.keys())
plt.ylim(0, 100)
plt.title('Pruning per Layer (from the Final Pareto Front)')
plt.ylabel('Filters Pruned (%)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save and show the plot
plt.savefig('pruning_percentages.png', dpi=600)
plt.show()