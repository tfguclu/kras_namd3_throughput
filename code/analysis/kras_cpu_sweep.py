"""
Core-count sweep: throughput density figure.

Run from KRAS_unbound_basarim/ (paths below point into KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/):

    python kras_cpu_sweep.py

Reads every step5_production*cpu.log in each sweep directory, takes the
per-interval ns/day from the TIMING records, and writes

    fig_cpu_sweep.pdf   sustained throughput against core count, one line per
                        stage, mean with standard-deviation error bars

The core count is read from the "<n>cpu" field of the file name. Logs marked
"gpuresoff" are skipped: GPU-resident off is a different execution path, not
another point in the core-count sweep.

The first timing record of each log is discarded as a cold-start value.
Nothing is written to the log directory.
"""

import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except ImportError:
    HAVE_KDE = False

# ---- settings ---------------------------------------------------------------
# one entry per optimization stage; each gets its own line in the figure
SWEEP_DIRS = {"KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_0": "baseline",
              "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_1": "default pairlist"}
LOG_GLOB = "step5_production*cpu.log"
# no GPU-resident benchmark exists at 20 cores for the baseline; its 20-core point
# is taken from the GPU-resident production log of the same configuration
EXTRA = {"KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_0": ("step5_production.log", 20)}
DROP_FIRST = 1

COL_W = 3.4
# stages are distinguished by line style and marker, not colour
STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]

plt.rcParams.update({
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

RE_TIMING = re.compile(r"^TIMING:\s+\d+.*?([\d.]+)\s+ns/days", re.M)
RE_PERF = re.compile(r"^PERFORMANCE:\s+\d+\s+averaging\s+([\d.]+)\s+ns/day",
                     re.M)
RE_CPU = re.compile(r"_(\d+)cpu\.log$")


def parse(path, ncpu=None):
    name = os.path.basename(path)
    m = RE_CPU.search(name)
    ncpu = ncpu or (int(m.group(1)) if m else None)
    text = open(path, errors="replace").read()
    # GPU-resident only if NAMD was launched with --GPUresident on
    resident = "Running with GPU-resident mode" in text
    rates = RE_TIMING.findall(text)
    kind = "TIMING"
    if not rates:
        rates = RE_PERF.findall(text)
        kind = "PERFORMANCE"
    rates = [float(x) for x in rates[DROP_FIRST:]]
    if not rates:
        print(f"# {name}: no timing records, skipped")
        return None

    label = (f"{ncpu} CPU" if ncpu else name)
    if not resident:
        label += ", GPU-res off"
    return {"name": name, "ncpu": ncpu or 0, "resident": resident,
            "label": label, "rate": np.asarray(rates), "kind": kind}


def density(ax, x, color, lo, hi, lw=1.4, ls="-", fill=False):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    grid = np.linspace(lo, hi, 500)
    if HAVE_KDE and x.size > 5 and np.ptp(x) > 0:
        y = gaussian_kde(x)(grid)
    else:
        y, edges = np.histogram(x, bins=60, range=(lo, hi), density=True)
        grid = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(grid, y, color=color, lw=lw, ls=ls)
    if fill:
        ax.fill_between(grid, y, color=color, alpha=0.13, lw=0)


if __name__ == "__main__":
    if not HAVE_KDE:
        print("# scipy not found -- histograms instead of KDE")

    stages = []
    for d, lab in SWEEP_DIRS.items():
        if not os.path.isdir(d):
            print(f"# {d}: not found, skipped")
            continue
        files = sorted(glob.glob(os.path.join(d, LOG_GLOB)))
        if not files:
            print(f"# {d}: no log matched {LOG_GLOB}")
            continue

        runs = [r for r in (parse(f) for f in files) if r]
        if d in EXTRA:
            f, n = EXTRA[d]
            r = parse(os.path.join(d, f), n)
            if r and not any(x["ncpu"] == n and x["resident"] for x in runs):
                runs.append(r)
        for r in runs:
            if not r["resident"]:
                print(f"# {d}: skipped (GPU-resident off): {r['name']}")
        runs = [r for r in runs if r["resident"] and r["ncpu"]]
        if not runs:
            continue
        runs.sort(key=lambda r: r["ncpu"])

        print(f"=== {lab}  ({d})")
        for r in runs:
            print(f"  {r['ncpu']:>3} CPU  {len(r['rate']):>5} records  "
                  f"mean {r['rate'].mean():7.1f}  sd {r['rate'].std():5.1f}  "
                  f"median {np.median(r['rate']):7.1f} ns/day")
        print()
        stages.append((lab, runs))

    if not stages:
        raise SystemExit("no usable logs found")

    # core counts, as the union across stages, in ascending order
    cores = sorted({r["ncpu"] for _, runs in stages for r in runs})
    pos = {n: i for i, n in enumerate(cores)}

    fig, ax = plt.subplots(figsize=(COL_W, 2.5), constrained_layout=True)
    lo_all, hi_all = [], []
    for k, (lab, runs) in enumerate(stages):
        x = np.array([r["ncpu"] for r in runs], dtype=float)
        y = np.array([r["rate"].mean() for r in runs])
        e = np.array([r["rate"].std() for r in runs])
        ls, mk = STYLES[k % len(STYLES)]
        ax.errorbar(x, y, yerr=e, color="k", ecolor="k", ls=ls, marker=mk,
                    markerfacecolor="k", elinewidth=1.0, capsize=3,
                    capthick=1.0, markersize=4.5, lw=1.2, zorder=3,
                    label=lab if len(stages) > 1 else None)
        lo_all.append((y - e).min())
        hi_all.append((y + e).max())

    ax.set_xticks(cores)
    ax.set_xticklabels([str(n) for n in cores])
    ax.set_xlim(0, max(cores) + 2)

    ylo = np.floor((min(lo_all) - 2) / 10.0) * 10
    yhi = np.ceil((max(hi_all) + 2) / 10.0) * 10
    ax.set_ylim(ylo, yhi)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.set_xlabel("CPU cores per GPU", fontsize=8.5)
    ax.set_ylabel("throughput (ns/day)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    if len(stages) > 1:
        ax.legend(fontsize=8, loc="best")

    for ext in ("pdf", "png"):

        fig.savefig(f"fig_cpu_sweep.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_cpu_sweep.pdf")