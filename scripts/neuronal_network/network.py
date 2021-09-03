import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow
import numpy as np

layers = (4, 7, 7, 3)


x_dist = 10
y_dist = 5

kwargs = {"edgecolor": "#a7a7a7", "linewidth": 1, "zorder": 10}


def is_activated():
    return np.random.rand() < 0.6


fig = plt.figure(figsize=(5, 4), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1)

network_pos = []
x_pos = 3
for ith_layer, neurons in enumerate(layers):
    pos = []
    for ith_neuron in range(neurons):
        y_pos = -y_dist * neurons / 2 + y_dist * (ith_neuron + 0.5)
        pos.append((x_pos, y_pos))
    x_pos += x_dist
    network_pos.append(pos)


for ith_layer, layer_pos in enumerate(network_pos):
    for neuron_pos in layer_pos:

        kwargs["facecolor"] = "#e2e2e2"
        if (ith_layer != 0) and (ith_layer != len(layers) - 1):
            if not is_activated():
                kwargs["facecolor"] = "#f6f6f6"

        circle = Circle(neuron_pos, 1.5, **kwargs)
        ax.add_patch(circle)

kwargs = {"color": "#ececec", "zorder": 1}
for layer_current, layer_next in zip(network_pos[:-1], network_pos[1:]):
    for neuron_current in layer_current:
        for neuron_next in layer_next:
            plt.plot(
                [neuron_current[0], neuron_next[0]],
                [neuron_current[1], neuron_next[1]],
                **kwargs
            )

# x, y = (x_dist * (len(layers) - 1) + 0.5, -y_dist * layers[-1] / 2)
# width, height = 5, y_dist * layers[-1]
# kwargs = {"fill": False, "edgecolor": "#b1b1b1", "zorder": 2}
# rec = Rectangle((x, y), width, height, **kwargs)
# ax.add_patch(rec)

# for neuron_pos in network_pos[-1]:
#     arr = Arrow(neuron_pos[0] + 4, neuron_pos[1], 3, 0, **kwargs)
#     ax.add_patch(arr)

ax.set_xlim(0, (len(layers) -1) * x_dist +6)
ylim = y_dist * np.max(layers) / 2 + 1
ax.set_ylim(-ylim, ylim)
ax.set_aspect(True)
ax.set_axis_off()

plt.savefig("network.jpg", dpi=300, bbox_inches="tight", pad_inches=0)
