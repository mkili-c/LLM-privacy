"""
Line plots of per-layer distances from a layer sweep CSV.

Usage:
    python plotting/plot_layers.py [csv] [out_prefix]
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    "../results/layer_sweep/qwen_layer_sweep_privacy_prompt_rewrites_floor_100107577_mean.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else "../figures/layers"

METRICS = [("udist2", "Unbiased Euclidian Distance of Answer Embedding Means to Original Mean"),
           ("dist2", "Biased Euclidian Distance of Answer Embedding Means to Original Mean")]

with open(CSV) as f:
    rows = list(csv.DictReader(f))

layers = [int(r["layer"]) for r in rows]

for column, label in METRICS:
    fig, ax = plt.subplots(figsize=(12, 7))
    for kind, color in [("para", "#2a78d6"), ("priv", "#eb6834")]:
        ax.plot(layers, [float(r[f"{column}_{kind}"]) for r in rows],
                linestyle="-", color=color, linewidth=2.0,
                marker="o", markersize=4,
                label="paraphrase" if kind == "para" else "privatized")
    
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("layer")
    ax.set_xticks(layers)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"{column} by layer and rewrite type")
    fig.tight_layout()
    out = f"{OUT}_{column}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


fig_m, ax_m = plt.subplots(figsize=(12, 7))
for column, _ in METRICS:
    ax_m.plot(layers, [float(r[f"{column}_mean_diff (gap priv-para)"]) for r in rows],
                        linestyle="-", color="red" if column == "udist2" else "green", linewidth=2.0,
                        marker="o", markersize=4,
                        label=f"{column} mean gap")

ax_m.legend(loc="upper left", fontsize=9)
ax_m.set_xlabel("layer")
ax_m.set_xticks(layers)
ax_m.tick_params(axis="x", labelsize=7)
ax_m.set_ylabel(label, fontsize=9)
ax_m.set_ylim(bottom=0.0)
ax_m.set_title(f"mean gaps by layer and rewrite type")
fig_m.tight_layout()
out = f"{OUT}_mean_gaps.png"
fig_m.savefig(out, dpi=200)
plt.close(fig_m)
print(f"wrote {out}")


fig_m, ax_m = plt.subplots(figsize=(12, 7))
for column, _ in METRICS:
    ax_m.plot(layers, [float(r[f"{column}_rank"]) for r in rows],
                        linestyle="-", color="red" if column == "udist2" else "green", linewidth=2.0,
                        marker="o", markersize=4,
                        label=f"{column} rank")

ax_m.legend(loc="upper left", fontsize=9)
ax_m.set_xlabel("layer")
ax_m.set_xticks(layers)
ax_m.tick_params(axis="x", labelsize=7)
ax_m.set_ylabel(label, fontsize=9)
ax_m.set_ylim(bottom=60.0)
ax_m.set_title(f"ranks by layer and rewrite type")
fig_m.tight_layout()
out = f"{OUT}_ranks.png"
fig_m.savefig(out, dpi=200)
plt.close(fig_m)
print(f"wrote {out}")



fig_m, ax_m = plt.subplots(figsize=(12, 7))
for type in ("para", "priv"):
    ax_m.plot(layers, [float(r[f"cross_{type}"]) for r in rows],
                        linestyle="-", color="#eb6834" if type == "para" else "purple", linewidth=2.0,
                        marker="o", markersize=4,
                        label=f"{type} mean pairwise cos")

ax_m.legend(loc="upper left", fontsize=9)
ax_m.set_xlabel("layer")
ax_m.set_xticks(layers)
ax_m.tick_params(axis="x", labelsize=7)
ax_m.set_ylabel(label, fontsize=9)
ax_m.set_ylim(bottom=0.96)
ax_m.set_title(f"mean pairwise cosine similarity by layer and rewrite type")
fig_m.tight_layout()
out = f"{OUT}_cos_sim.png"
fig_m.savefig(out, dpi=200)
plt.close(fig_m)
print(f"wrote {out}")