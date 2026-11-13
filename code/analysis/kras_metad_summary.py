#!/usr/bin/env python3
"""
Switch I sampling and throughput for the KRAS WT-metaD replicates.

Run from KRAS_unbound_basarim/:
    python kras_metad_summary.py

Per replicate, reads every stepN_*_metadynamics_1d piece in <rep>/namd/
(.BAK ignored, pieces ordered by N):
  *.colvars.traj -> Switch I CV (Ser17-Gln25-Thr35 CA angle)
  *.log          -> ns/day from PERFORMANCE lines

Writes
  fig_metad.(png|pdf)            CV vs time per replicate (frames after CV~0 lock-in
                                 in grey) and, below, P(CV) over pre-trap frames
  fig_metad_throughput.(png|pdf) median ns/day per replicate vs unbiased baseline
and prints a summary table.
"""
import glob, os, re
import numpy as np
import matplotlib.pyplot as plt

# ---------------- settings ----------------
REPS = {
    "rep1": "KRAS_unbound_mg_gtp_metaD/namd",
    "rep2": "KRAS_unbound_mg_gtp_metaD_2/namd",
    "rep3": "KRAS_unbound_mg_gtp_metaD_4/namd",
}
COLORS = {"rep1": "#1f6fb4", "rep2": "#c8571f", "rep3": "#3f8f5c"}   # same as the other figures
COL_W = 3.4
CV_COLUMN = None            # None = first column after 'step'
CLOSED, OPEN = 45.0, 90.0   # deg, as in the paper
BASELINE_NSDAY = 147.0      # unbiased CHARMM-GUI baseline
HOLD_NS = 5.0               # an opening must stay above OPEN this long
PIECE_GLOB = "step*_metadynamics_1d"
FS_TO_NS = 1e-6             # 1 fs = 1e-6 ns
TRAP_CV, TRAP_MIN_NS = 20.0, 50.0   # corner-trap detection
plt.rcParams.update({            # identical to the other kras_*.py figures
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# ------------------------------------------

def pieces(d):
    pref = {f[:-4] for f in glob.glob(os.path.join(d, PIECE_GLOB + ".log"))}
    return sorted(pref, key=lambda p: int(re.search(r"step(\d+)_", os.path.basename(p)).group(1)))

def timestep_fs(log):
    with open(log, errors="ignore") as fh:
        for line in fh:
            m = re.search(r"Info: TIMESTEP\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"no TIMESTEP in {log}")

def read_colvars(traj, col):
    names, rows = None, []
    with open(traj, errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                names = names or line[1:].split()
                continue
            parts = line.split()
            if names and len(parts) >= len(names):
                try:
                    rows.append([float(x) for x in parts[:len(names)]])
                except ValueError:
                    pass
    if not rows:
        return np.empty(0), np.empty(0), None
    a = np.array(rows)
    icv = names.index(col) if col else 1
    return a[:, 0], a[:, icv], names[icv]

def read_perf(log):
    """Instantaneous ns/day: 'current X ns/day' if NAMD prints it, else the
    running average. First record of each log (warm-up) dropped."""
    cur, avg = [], []
    with open(log, errors="ignore") as fh:
        for line in fh:
            if "PERFORMANCE:" not in line:
                continue
            m = re.search(r"current\s+([\d.]+)\s+ns/day", line)
            if m: cur.append(float(m.group(1)))
            m = re.search(r"averaging\s+([\d.]+)\s+ns/day", line)
            if m: avg.append(float(m.group(1)))
    v = cur if cur else avg
    return v[1:], ("current" if cur else "averaging")

def load_rep(d):
    steps, cv, perf = [], [], []
    dt = cvname = kind = None
    for p in pieces(d):
        dt = dt or timestep_fs(p + ".log")
        if os.path.exists(p + ".colvars.traj"):
            s, c, cvname = read_colvars(p + ".colvars.traj", CV_COLUMN)
            steps.append(s); cv.append(c)
        v, kind = read_perf(p + ".log")
        perf += v
    s = np.concatenate(steps); c = np.concatenate(cv)
    s, idx = np.unique(s, return_index=True)          # drop restart overlaps
    t_ns = (s - s[0]) * dt * FS_TO_NS                  # time from start of metaD
    return dict(t=t_ns, cv=c[idx], perf=np.array(perf), dt=dt, cvname=cvname, kind=kind)

def first_open(t, cv):
    above = cv > OPEN
    i = 0
    while i < len(cv):
        if above[i]:
            j = i
            while j < len(cv) and above[j]:
                j += 1
            if t[j - 1] - t[i] >= HOLD_NS:
                return t[i]
            i = j
        else:
            i += 1
    return None

def trap_start(t, cv):
    """Start of the final stretch in which CV never exceeds TRAP_CV (the
    corner trap at CV ~ 0). None if that stretch is shorter than TRAP_MIN_NS."""
    above = np.where(cv >= TRAP_CV)[0]
    i = above[-1] + 1 if len(above) else 0
    if i < len(t) and t[-1] - t[i] >= TRAP_MIN_NS:
        return t[i]
    return None

def shade(ax, vertical=True):
    f = ax.axvspan if vertical else ax.axhspan
    f(0, CLOSED, color="#eef1f4", lw=0, zorder=0); f(OPEN, 180, color="#e2e6ea", lw=0, zorder=0)

def smooth_density(x, bins, sigma_bins=1.5):
    h, e = np.histogram(x, bins=bins, density=True)
    k = np.exp(-0.5 * (np.arange(-4, 5) / sigma_bins) ** 2); k /= k.sum()
    return 0.5 * (e[1:] + e[:-1]), np.convolve(h, k, mode="same")

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    data = {r: load_rep(d) for r, d in REPS.items() if os.path.isdir(d)}
    for r, d in REPS.items():
        if r not in data: print(f"WARNING: {d} not found")
    for D in data.values():
        D["trap"] = trap_start(D["t"], D["cv"])
        D["use"] = D["t"] < D["trap"] if D["trap"] is not None else np.ones(len(D["t"]), bool)

    # ---- fig_metad: CV trace per replicate (trapped part grey) + P(CV) below ----
    n = len(data)
    fig = plt.figure(figsize=(COL_W, 1.15 * n + 2.1), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1] * n + [1.6])
    ax0 = None
    for i, (r, D) in enumerate(data.items()):
        ax = fig.add_subplot(gs[i], sharex=ax0)
        ax0 = ax0 or ax
        shade(ax, vertical=False)
        t, cv, u = D["t"] / 1000, D["cv"], D["use"]
        ax.plot(t[u], cv[u], color=COLORS[r], lw=0.4)
        if D["trap"] is not None:
            ax.plot(t[~u], cv[~u], color="0.6", lw=0.4)
        ax.set_ylim(0, 180); ax.set_yticks([0, 90, 180])
        ax.set_ylabel("CV (°)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.text(0.985, 0.92, r, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=COLORS[r])
        if i == n - 1:
            ax.set_xlabel("simulated time (µs)", fontsize=8.5)
        else:
            ax.tick_params(labelbottom=False)
    bins = np.arange(0, 182.5, 2.5)
    ax = fig.add_subplot(gs[n])
    shade(ax)
    for r, D in data.items():
        x, y = smooth_density(D["cv"][D["use"]], bins)
        ax.fill_between(x, y, color=COLORS[r], alpha=0.2, lw=0)
        ax.plot(x, y, color=COLORS[r], lw=1.0, label=r)
    ax.set_xlim(0, 180); ax.set_ylim(bottom=0); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Switch I CV (°)", fontsize=8.5); ax.set_ylabel("density", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_metad")

    # ---- throughput: medians vs unbiased baseline ----
    labels = ["unbiased MD"] + [f"metaD {r}" for r in data]
    vals = [BASELINE_NSDAY] + [np.median(D["perf"]) if len(D["perf"]) else np.nan for D in data.values()]
    cols = ["0.35"] + [COLORS[r] for r in data]
    fig, ax = plt.subplots(figsize=(COL_W, 1.5), constrained_layout=True)
    yy = np.arange(len(vals))[::-1]
    ax.barh(yy, vals, color=cols, height=0.6)
    for y, v in zip(yy, vals):
        ax.text(v + 2, y, f"{v:.0f}", va="center", fontsize=7.5)
    ax.set_yticks(yy); ax.set_yticklabels(labels)
    ax.set_xlim(0, max(vals) * 1.15); ax.set_xlabel("throughput (ns/day), 1 H100, 2 fs")
    save(fig, "fig_metad_throughput")

    # ---- summary ----
    print(f"{'rep':5} {'len(us)':>8} {'%closed':>8} {'%open':>6} {'trap(us)':>9} "
          f"{'1st open(ns)':>12} {'ns/day':>7}   (% over pre-trap frames)")
    for r, D in data.items():
        cv, fo = D["cv"][D["use"]], first_open(D["t"], D["cv"])
        trap = "none" if D["trap"] is None else "%.2f" % (D["trap"] / 1000)
        fos = "none" if fo is None else "%.0f" % fo
        med = np.median(D["perf"]) if len(D["perf"]) else float("nan")
        print(f"{r:5} {D['t'][-1]/1000:8.2f} {100*np.mean(cv<CLOSED):8.1f} {100*np.mean(cv>OPEN):6.1f} "
              f"{trap:>9} {fos:>12} {med:7.1f}")
    D0 = next(iter(data.values()))
    print(f"CV column: {D0['cvname']}; ns/day from '{D0['kind']}' values")

if __name__ == "__main__":
    main()
