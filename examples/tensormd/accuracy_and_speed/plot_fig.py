import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

dr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

static = np.loadtxt("compute/rmse.txt")
dynamic = np.loadtxt("thermo.txt", skiprows=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

ax = axes[0]
ax.plot(dr, static, "ro-", label="Force (eV/$\mathrm{\AA}$)")
ax.set_title("Static", fontsize=14)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=10, frameon=False)
ax.set_xlabel(r"$\Delta r$ ($\mathrm{\AA}$)", fontsize=14)
ax.set_ylabel(r"RMSE", fontsize=14)
ax.text(-0.12, 1.05, "a)", transform=ax.transAxes, fontsize=12, fontweight="bold")

ax = axes[1]
ax.set_title("Dynamic", fontsize=14)
ax.plot(dr, dynamic[:, 1], "ko-", label="Temperature (K)")
ax.plot(dr, np.maximum(1e-8, dynamic[:, 3] / 1024), "go-", label="Potential Energy (eV/atom)")
ax.plot(dr, dynamic[:, 5], "bo-", label="Press (bar)")
ax.legend(fontsize=10, frameon=False)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\Delta r$ ($\mathrm{\AA}$)", fontsize=14)
ax.set_ylabel(r"RMSE", fontsize=14)
ax.text(-0.12, 1.05, "b)", transform=ax.transAxes, fontsize=12, fontweight="bold")

# Loop time from the speed directory
loop_times = np.asarray([
    627.13,         # fnn
    107.342,        # dr = 0.1
    107.399,        # dr = 0.05
    107.511,        # dr = 0.01
    107.761,        # dr = 0.005
    108.243,        # dr = 0.001
    112.282,        # dr = 0.0005
    125.636         # dr = 0.0001
])

ratio = loop_times[0] / loop_times[1:]

ax = axes[2]

ax.plot(dr, ratio, "k^-")
ax.set_xlabel(r"$\Delta r$ ($\mathrm{\AA}$)", fontsize=14)
ax.set_ylabel(r"Speedup", fontsize=14)
ax.set_xscale("log")
ax.set_ylim([4.5, 6.0])
ax.text(-0.12, 1.05, "c)", transform=ax.transAxes, fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("accuracy_speed.png", dpi=300)

