import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def calc_segements(track):
    pos = np.stack((track.start_x, track.start_y, track.start_z), axis=1)
    end = np.stack((track.end_x, track.end_y, track.end_z), axis=1)
    return np.stack((pos, end), axis=1)


def get_color(pdgs, p_colors):
    colors = np.full(*pdgs.shape, p_colors[None])
    for pdg, color in p_colors.items():
        colors[pdgs == pdg] = color
    return colors


df = pd.read_parquet("build/em_shower_outputs/tracks/tracks.parquet")

segments = calc_segements(df)
segments2d = segments[:, :, 1:3]

p_colors = {
    None: "C0",
    -11: "C1",
    11: "C2",
    22: "C3",
}

fig = plt.figure(figsize=(4, 4), dpi=300, constrained_layout=True)
ax = fig.add_subplot(1, 1, 1)

col = LineCollection(
    segments2d, linewidths=0.2, colors=get_color(df.pdg, p_colors), alpha=1.0
)
col.set_rasterized(True)
ax.add_collection(col)


x_min, x_max = np.min(segments2d[:, :, 0]), np.max(segments2d[:, :, 0])
y_min, y_max = np.min(segments2d[:, :, 1]), np.max(segments2d[:, :, 1])


x = np.linspace(x_min, x_max, 30)
y = np.linspace(y_min, y_max, 30)

X, Y = np.meshgrid(x, y)


def atmosphere(x, y):
    return np.exp(y / 5.5e5)


Z2 = atmosphere(X, Y)

im2 = ax.imshow(
    Z2,
    cmap=plt.cm.Blues,
    alpha=0.4,
    interpolation="bilinear",
    extent=[-1e5, 1e5, y_min, y_max],
)

ax.set_xlim(-0.5e4, 1.5e4)
ax.set_ylim(y_min, y_max)
ax.set_axis_off()
# ax.set_aspect(False)
# ax.margins(0.5,0.5)
# plt.show()
plt.savefig("build/shower.jpg", transparent=False, bbox_inches = 'tight', pad_inches = 0)
