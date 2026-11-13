"""
Throughput analysis from NAMD logs for the KRAS HMR replicates.

Run from KRAS_unbound_basarim/ (paths below point into KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/):

    python kras_hmr_benchmark.py

For each replicate it finds the chained step5_production*.log files, reads the
per-interval ns/day from the TIMING records (falling back to the PERFORMANCE
running average if a log has no TIMING lines), prints a summary, and writes

    fig_throughput.pdf   sustained throughput per replicate and pooled density

The first timing record of each log is discarded: NAMD prints it before the
run reaches steady state, so it reports a cold-start rate.

Nothing is written to the replicate directories.
"""

import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except ImportError:
    HAVE_KDE = False

# ---- settings ---------------------------------------------------------------
DIRS = {"KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3": "rep1", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_2": "rep2", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_3": "rep3"}
LOG_GLOB = "step5_production*.log"

TIMESTEP_FS = 4.0        # HMR production timestep
DROP_FIRST = 1           # timing records to discard per log (cold start)

COLORS = ["#1f6fb4", "#c8571f", "#3f8f5c"]
POOLED_COLOR = "#333333"
COL_W = 3.4              # IEEE single-column width, inches

plt.rcParams.update({
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# NAMD writes e.g.
#   TIMING: 60000  CPU: 50.53, 0.000828/step  Wall: 50.19, 0.000833/step,
#           207.487 ns/days, 1.14 hours remaining, 0.0 MB of memory in use.
#   PERFORMANCE: 60000  averaging 206.996 ns/day, 0.000834797 sec/step ...
#   Info: Benchmark time: 20 CPUs 0.000577 s/step 299.28 ns/day 0 MB memory
RE_TIMING = re.compile(r"^TIMING:\s+(\d+).*?([\d.]+)\s+ns/days", re.M)
RE_PERF = re.compile(r"^PERFORMANCE:\s+(\d+)\s+averaging\s+([\d.]+)\s+ns/day",
                     re.M)
RE_BENCH = re.compile(r"Benchmark time:\s+(\d+)\s+CPUs\s+([\d.eE+-]+)\s+s/step"
                      r"\s+([\d.]+)\s+ns/day")


def order(path):
    """Base log first, then _1, _2, ... in numeric order."""
    m = re.search(r"_(\d+)\.log$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def read_logs(directory):
    files = sorted((f for f in glob.glob(os.path.join(directory, LOG_GLOB))
                    if not f.endswith("_HMR.log")),   # shared 100 ns benchmark copy, not production
                   key=order)
    if not files:
        raise SystemExit(f"no log matched {LOG_GLOB} in {directory}")

    steps, rate, piece, source = [], [], [], []
    bench = []
    for k, f in enumerate(files):
        text = open(f, errors="replace").read()
        hits = RE_TIMING.findall(text)
        kind = "TIMING"
        if not hits:
            hits = RE_PERF.findall(text)
            kind = "PERFORMANCE"
        if not hits:
            print(f"#   {os.path.basename(f)}: no timing records, skipped")
            continue
        hits = hits[DROP_FIRST:]
        if not hits:
            continue
        for st, ns in hits:
            steps.append(int(st))
            rate.append(float(ns))
            piece.append(k)
        source.append(kind)
        bench += [float(b[2]) for b in RE_BENCH.findall(text)]

    if not rate:
        raise SystemExit(f"no usable timing records in {directory}")

    return {
        "files": files,
        "steps": np.asarray(steps, dtype=np.int64),
        "rate": np.asarray(rate, dtype=np.float64),
        "piece": np.asarray(piece, dtype=np.int32),
        "kind": sorted(set(source)),
        "bench": np.asarray(bench, dtype=np.float64),
    }


def summarise(label, d):
    r = d["rate"]
    # simulated time covered, from the largest step index in each piece
    total_steps = 0
    for p in np.unique(d["piece"]):
        total_steps += d["steps"][d["piece"] == p].max()
    sim_ns = total_steps * TIMESTEP_FS / 1e6
    wall_h = sim_ns / np.median(r) * 24.0 if np.median(r) > 0 else np.nan

    print(f"--- {label}: {len(d['files'])} log(s), {len(r)} timing records "
          f"({', '.join(d['kind'])})")
    print(f"  ns/day  mean {r.mean():.1f}  median {np.median(r):.1f}  "
          f"sd {r.std():.1f}  min {r.min():.1f}  max {r.max():.1f}")
    print(f"  5th-95th percentile {np.percentile(r, 5):.1f} - "
          f"{np.percentile(r, 95):.1f}")
    print(f"  approx {sim_ns / 1000:.2f} us simulated, "
          f"{wall_h / 24:.1f} d of GPU time at the median rate")
    if d["bench"].size:
        print(f"  Benchmark time records: {np.median(d['bench']):.1f} ns/day "
              f"(n={d['bench'].size})")
    print()


def density(ax, x, color, lo, hi, lw=1.3, ls="-", fill=False, label=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    grid = np.linspace(lo, hi, 400)
    if HAVE_KDE and x.size > 5 and np.ptp(x) > 0:
        y = gaussian_kde(x)(grid)
    else:
        y, edges = np.histogram(x, bins=50, range=(lo, hi), density=True)
        grid = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(grid, y, color=color, lw=lw, ls=ls, label=label)
    if fill:
        ax.fill_between(grid, y, color=color, alpha=0.15, lw=0)


def figure(data):
    labels = list(data)
    pooled = np.concatenate([data[l]["rate"] for l in labels])
    lo = max(0.0, np.percentile(pooled, 0.5) - 20)
    hi = np.percentile(pooled, 99.5) + 20

    n = len(labels)
    fig = plt.figure(figsize=(COL_W, 1.15 * n + 2.1), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1.0] * n + [1.35])

    for i, lab in enumerate(labels):
        d = data[lab]
        c = COLORS[i % len(COLORS)]
        # cumulative simulated time on the x axis, so restarts line up in time
        starts, run = {}, 0
        for p in np.unique(d["piece"]):
            starts[p] = run
            run += d["steps"][d["piece"] == p].max()
        x = np.array([starts[p] for p in d["piece"]]) + d["steps"]
        x = x * TIMESTEP_FS / 1e6 / 1000.0        # -> microseconds

        ax = fig.add_subplot(gs[i])
        ax.plot(x, d["rate"], color=c, lw=0.5)
        med = np.median(d["rate"])
        ax.axhline(med, color="0.35", lw=0.7, ls="--")
        ax.text(0.985, 0.06, f"median {med:.0f} ns/day", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color="0.35",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0,
                          alpha=0.75))
        ax.set_ylim(lo, hi)
        ax.set_ylabel("ns/day", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.text(0.05, 0.92, lab, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, color=c)
        if i == n - 1:
            ax.set_xlabel("simulated time (\u00b5s)", fontsize=8.5)
        else:
            ax.set_xticklabels([])

    ax = fig.add_subplot(gs[n])
    for i, lab in enumerate(labels):
        density(ax, data[lab]["rate"], COLORS[i % len(COLORS)], lo, hi,
                label=lab)
    density(ax, pooled, POOLED_COLOR, lo, hi, lw=1.6, ls="--",
            label=f"pooled ({np.median(pooled):.0f})")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("throughput (ns/day)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.5, columnspacing=1.0, handletextpad=0.5)

    for ext in ("pdf", "png"):

        fig.savefig(f"fig_throughput.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_throughput.pdf")


if __name__ == "__main__":
    if not HAVE_KDE:
        print("# scipy not found -- histograms instead of KDE")

    data = {}
    for d, lab in DIRS.items():
        print(f"# {d}")
        data[lab] = read_logs(d)
    print()

    for lab in data:
        summarise(lab, data[lab])

    pooled = np.concatenate([data[l]["rate"] for l in data])
    print(f"=== pooled over {len(data)} replicates, {len(pooled)} records")
    print(f"  ns/day  mean {pooled.mean():.1f}  median {np.median(pooled):.1f}"
          f"  sd {pooled.std():.1f}")
    print(f"  5th-95th percentile {np.percentile(pooled, 5):.1f} - "
          f"{np.percentile(pooled, 95):.1f}")
    print()

    figure(data)
