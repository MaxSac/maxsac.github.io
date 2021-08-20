import numpy as np
import matplotlib.pyplot as plt
import proposal as pp
import logging
import sys
import pandas as pd
from dataclasses import make_dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


logger = logging.getLogger("track_plotter")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

pp.InterpolationSettings.tables_path = "/home/msackel/.cache/PROPOSAL"

targets = {"air": pp.medium.Air(), "standardrock": pp.medium.StandardRock()}

utilities = {}
for target in targets.values():

    prop_def = {
        "particle_def": pp.particle.MuMinusDef(),
        "cuts": pp.EnergyCutSettings(500, 1, False),
        "target": target,
        "interpolate": True,
    }

    cross = pp.crosssection.make_std_crosssection(**prop_def)
    collection = pp.PropagationUtilityCollection()
    collection.displacement = pp.make_displacement(cross, True)
    collection.interaction = pp.make_interaction(cross, True)
    collection.time = pp.make_time_approximate()

    utilities[f"{target.name}"] = pp.PropagationUtility(collection)

logger.info("utility build")


earth_radius = 6300 * 1e5  # km -> cm
atmosphere_depth = 2000 * 1e5  # km -> cm

geometries = {}

geometries["air"] = pp.geometry.Sphere(pp.Cartesian3D(0, 0, 0), np.Infinity)
geometries["air"].hierarchy = 1
geometries["standardrock"] = pp.geometry.Sphere(
    pp.Cartesian3D(0, 0, 0), earth_radius
)
geometries["standardrock"].hierarchy = 2


profiles = {}

axis = pp.density_distribution.radial_axis(pp.Cartesian3D(0, 0, 0))
sigma = -5.5 * 1e5  # km -> cm
# profiles["air"] = pp.density_distribution.density_exponential(axis, sigma, earth_radius, targets["air"].mass_density)
profiles["air"] = pp.density_distribution.density_homogeneous(
    1e-1 * targets["air"].mass_density
)

profiles["standardrock"] = pp.density_distribution.density_homogeneous(
    1e-4 * targets["standardrock"].mass_density
)

env = []
for utility, geometry, prof in zip(
    utilities.values(), geometries.values(), profiles.values()
):
    env.append((geometry, utility, prof))

prop = pp.Propagator(pp.particle.MuMinusDef(), env)


def get_injection_point():
    phi = 0
    theta = 2 * np.pi * np.random.random()
    pos = pp.Spherical3D(earth_radius + atmosphere_depth, phi, theta)
    return pp.Cartesian3D(pos)


def get_direction(pos):
    pos = pp.Spherical3D(-pos)
    pos.radius = 1

    rnd1 = np.random.normal(0, np.pi / 20)
    rnd2 = np.random.normal(0, np.pi / 20)

    pos.azimuth = pos.azimuth + rnd1
    pos.zenith = pos.zenith + rnd2

    return pp.Cartesian3D(pos)


Loss = make_dataclass(
    "Loss",
    [
        ("E", np.float32),
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("time", np.float32),
        ("type", int),
    ],
)


def produce_losses():
    state = pp.particle.ParticleState()
    state.type = pp.particle.Particle_Type.MuMinus
    state.energy = 0.5e6  # MeV
    state.position = get_injection_point()
    state.direction = get_direction(state.position)

    sec = prop.propagate(state, min_energy=1000)

    losses = [
        Loss(l.energy, l.position.x, l.position.y, l.position.z, l.time, l.type)
        for l in sec.stochastic_losses()
    ]
    return pd.DataFrame(losses)


def produce_data():
    time = 0
    data = []
    for i in range(200):
        losses = produce_losses()
        halftime = 5e-3
        time += np.random.exponential(halftime)
        losses.time += time
        data.append(losses)
    return pd.concat(data, ignore_index=True)


class UpdateDist:
    def __init__(self, ax, data):
        self.ax = ax
        self.data = data
        outer_radius = earth_radius + atmosphere_depth
        oversized_radius = 1.1 * outer_radius

        p1 = Circle((0, 0), earth_radius, alpha=0.10, color="C0")
        p2 = Circle((0, 0), outer_radius, alpha=0.05, color="C1")
        self.ax.add_artist(p1)
        self.ax.add_artist(p2)

        self.ax.set_xlim(oversized_radius, -oversized_radius)
        self.ax.set_ylim(oversized_radius, -oversized_radius)

        self.paths = self.ax.scatter(self.data.x, self.data.z, alpha=0.6)

        self.n_dots = self.data.index[-1] + 1
        self.colors = np.array(["C0"] * self.n_dots)

        for i, t in enumerate(np.unique(self.data.type)):
            mask = self.data.type == np.full((self.n_dots), t)
            self.colors[mask] = f"C{i}"
        self.paths.set_edgecolors(self.colors)
        self.paths.set_facecolors(self.colors)

        self.time_delta = 3e-3
        self.time_step = 5e-6
        self.time = -self.time_delta

        self.size = np.log(self.data.E) ** 5 / 500

    def __call__(self, i):
        self.time += i * self.time_step

        cond1 = self.data.time > self.time
        cond2 = self.data.time < self.time + self.time_delta
        timeslot = cond1 & cond2

        self.paths.set_sizes(timeslot * self.size)
        return self.paths


from matplotlib.animation import FuncAnimation

data = produce_data()
data.time -= data.time.min()

fig, ax = plt.subplots(dpi=200, constrained_layout=True)
ax.set_aspect(True)
ax.set_axis_off()

ud = UpdateDist(ax, data)
anim = FuncAnimation(fig, ud, frames=600, interval=100)
anim.save("videos/muon.mp4")
