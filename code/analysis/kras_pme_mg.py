"""
PME interpolation order 4: Mg(2+) coordination and throughput.

Run from KRAS_unbound_basarim/ (paths below point into KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/):

    python kras_pme_mg.py

These runs test whether reducing PMEInterpOrder from 6 to 4 perturbs the Mg
first shell. They use the reference integrator settings (2 fs, dcdfreq 50000,
so 100 ps per frame) and are 100 ns long, so they are short by design: a mesh
error would show up within nanoseconds or not at all.

For each replicate it
  1. reads the 100 ns PME-4 trajectory and computes the Mg first-shell
     composition, per-ligand Mg-O distances and octahedral angles,
  2. reads the matching log and reports throughput and wall-clock time,
and writes

    fig_pme_mg_shell.pdf   shell densities, one curve per replicate

No Switch I analysis: 100 ns is far below the first-passage time for that
transition. Nothing is written to the replicate directories.
"""

import os
import re
import glob

import numpy as np
import rawio
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
DIRS = {"KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_2": "rep1", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_2_2": "rep2", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_2_3": "rep3"}
PSF = "step3_input.psf"
DCD_GLOB = "step5_production.dcd"
LOG_GLOB = "step5_production_100ns_4pme.log"

TIMESTEP_FS = 2.0
DCDFREQ = 50000
FRAME_PS = TIMESTEP_FS * DCDFREQ / 1000.0      # 100 ps per frame
DROP_FIRST = 1                                 # cold-start timing records

MG_SHELL_CUT = 2.6
TRANS_MIN = 135.0

LIGANDS = ["SER17-OG", "THR35-OG1", "GTP-O2B", "GTP-O2G"]
COLORS = ["#1f6fb4", "#c8571f", "#3f8f5c"]
COL_W = 3.4

plt.rcParams.update({
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

RE_TIMING = re.compile(
    r"^TIMING:\s+(\d+)\s+CPU:\s+[\d.eE+-]+,.*?Wall:\s+([\d.eE+-]+),"
    r".*?([\d.]+)\s+ns/days", re.M)
RE_WALLCLOCK = re.compile(r"^WallClock:\s+([\d.eE+-]+)", re.M)
RE_BENCH = re.compile(r"Benchmark time:\s+\d+\s+CPUs\s+[\d.eE+-]+\s+s/step"
                      r"\s+([\d.]+)\s+ns/day")


# ---- trajectory -------------------------------------------------------------
def analyse(directory):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_angles, calc_bonds, distance_array

    dcds = sorted(glob.glob(os.path.join(directory, DCD_GLOB)))
    if not dcds:
        raise SystemExit(f"no DCD matched {DCD_GLOB} in {directory}")
    top = os.path.join(directory, PSF)
    if not os.path.isfile(top):
        raise SystemExit(f"missing topology: {top}")

    u = mda.Universe(top, dcds)

    def one(sel, what):
        g = u.select_atoms(sel)
        if g.n_atoms == 0:
            raise SystemExit(f"no atom for {what} ({sel}) in {directory}")
        return g[0]

    mg = one("resname MG and name MG", "Mg ion")
    named = {}
    for key, sel in (("SER17-OG", "resid 17 and name OG"),
                     ("THR35-OG1", "resid 35 and name OG1"),
                     ("GTP-O2B", "resname GTP and name O2B"),
                     ("GTP-O2G", "resname GTP and name O2G")):
        g = u.select_atoms(sel)
        if g.n_atoms:
            named[key] = g[0]
        else:
            print(f"#   WARNING: {key} not found ({sel})")

    water = u.select_atoms("(resname TIP3 HOH WAT SOL) and name OH2 OW O")
    pool = u.select_atoms("name O* and not name OH2 OW") + water
    site = u.atoms[np.array([a.index for a in named.values()])]

    print(f"# {directory}: {len(dcds)} DCD, topology {PSF}, "
          f"{FRAME_PS:.0f} ps per frame")
    print(f"#   ion {mg.resname}{mg.resid}-{mg.name}, "
          f"oxygen pool {pool.n_atoms}")

    keys = list(named)
    cn, nwat, site_d, cis, trans, dists = [], [], [], [], [], []

    for ts in u.trajectory:
        box = ts.dimensions
        d = distance_array(mg.position.reshape(1, 3), pool.positions, box=box)[0]
        sel = np.where(d < MG_SHELL_CUT)[0]
        cn.append(len(sel))
        nwat.append(int(np.isin(pool[sel].indices, water.indices).sum()))
        dists.append([calc_bonds(mg.position, named[k].position, box=box)
                      for k in keys])

        if len(sel) >= 2:
            pos = pool[sel].positions
            ai, bi = np.triu_indices(len(pos), k=1)
            centre = np.repeat(mg.position.reshape(1, 3), len(ai), axis=0)
            a = np.degrees(calc_angles(pos[ai], centre, pos[bi], box=box))
            c_, t_ = a[a < TRANS_MIN], a[a >= TRANS_MIN]
            cis.append(c_.mean() if c_.size else np.nan)
            trans.append(t_.mean() if t_.size else np.nan)
        else:
            cis.append(np.nan)
            trans.append(np.nan)

        site_d.append(distance_array(mg.position.reshape(1, 3),
                                     site.positions, box=box)[0].min())

    dists = np.asarray(dists, dtype=np.float32)
    return {
        "n": len(cn),
        "cn": np.asarray(cn, dtype=np.int16),
        "nwat": np.asarray(nwat, dtype=np.int16),
        "cis": np.asarray(cis, dtype=np.float32),
        "trans": np.asarray(trans, dtype=np.float32),
        "site": np.asarray(site_d, dtype=np.float32),
        "dist": {k: dists[:, j] for j, k in enumerate(keys)},
    }


def summarise_shell(label, r):
    ns = r["n"] * FRAME_PS / 1000.0
    print(f"--- {label}: {r['n']} frames, {ns:.0f} ns")
    print(f"  CN mean {r['cn'].mean():.3f}, CN==6 in "
          f"{100 * np.mean(r['cn'] == 6):.1f}% of frames, "
          f"waters {r['nwat'].mean():.2f}")
    for k in LIGANDS:
        if k not in r["dist"]:
            continue
        x = r["dist"][k]
        xi = x[x < MG_SHELL_CUT]
        print(f"    {k:<11} {xi.mean():.3f} +/- {xi.std():.3f} A   "
              f"occupancy {100 * np.mean(x < MG_SHELL_CUT):5.1f}%")
    print(f"  Mg to nearest protein/GTP oxygen {r['site'].mean():.3f} A "
          f"(max {r['site'].max():.3f})")
    print(f"  cis {np.nanmean(r['cis']):.2f} +/- {np.nanstd(r['cis']):.2f}, "
          f"trans {np.nanmean(r['trans']):.2f} +/- "
          f"{np.nanstd(r['trans']):.2f} deg")
    print()


# ---- logs -------------------------------------------------------------------
def dhm(seconds):
    if seconds is None:
        return "--"
    m = int(round(seconds / 60.0))
    d, rem = divmod(m, 1440)
    h, mi = divmod(rem, 60)
    return f"{d}:{h:02d}:{mi:02d}"


def read_log(directory):
    files = sorted(glob.glob(os.path.join(directory, LOG_GLOB)))
    if not files:
        print(f"#   no log matched {LOG_GLOB} in {directory}")
        return None
    text = open(files[0], errors="replace").read()
    hits = RE_TIMING.findall(text)
    if not hits:
        print(f"#   no TIMING records in {os.path.basename(files[0])}")
        return None
    hits = hits[DROP_FIRST:]
    steps = np.array([int(h[0]) for h in hits])
    wall = np.array([float(h[1]) for h in hits])
    rate = np.array([float(h[2]) for h in hits])

    wc = RE_WALLCLOCK.findall(text)
    total_wall = float(wc[-1]) if wc else float(wall[-1])
    bench = [float(b) for b in RE_BENCH.findall(text)]
    return {"file": os.path.basename(files[0]), "steps": steps, "rate": rate,
            "wall": total_wall, "exact": bool(wc), "bench": bench}


def summarise_log(label, g):
    if g is None:
        return
    sim_ns = g["steps"].max() * TIMESTEP_FS / 1e6
    eff = sim_ns / (g["wall"] / 86400.0) if g["wall"] else float("nan")
    print(f"--- {label}: {g['file']}, {len(g['rate'])} timing records")
    print(f"  ns/day  mean {g['rate'].mean():.1f}  median "
          f"{np.median(g['rate']):.1f}  sd {g['rate'].std():.1f}  "
          f"min {g['rate'].min():.1f}  max {g['rate'].max():.1f}")
    print(f"  {sim_ns:.1f} ns simulated in {dhm(g['wall'])} (d:hh:mm)"
          f"{'' if g['exact'] else '  [from last TIMING record]'}")
    print(f"  effective {eff:.1f} ns/day end to end")
    if g["bench"]:
        print(f"  Benchmark time records: {np.median(g['bench']):.1f} ns/day "
              f"(n={len(g['bench'])})")
    print()


# ---- figure -----------------------------------------------------------------
def density(ax, x, color, lo, hi, fill=False):
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
    ax.plot(grid, y, color=color, lw=1.3)
    if fill:
        ax.fill_between(grid, y, color=color, alpha=0.15, lw=0)


def figure_shell(reps):
    fig = plt.figure(figsize=(COL_W, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    for k, lig in enumerate(LIGANDS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        for i, (lab, r) in enumerate(reps.items()):
            if lig not in r["dist"]:
                continue
            x = r["dist"][lig]
            density(ax, x[x < MG_SHELL_CUT], COLORS[i % len(COLORS)],
                    1.6, 2.6, fill=True)
        ax.set_xlabel(f"Mg\u2013O, {lig} (\u00c5)", fontsize=8.5)
        ax.set_ylabel("density", fontsize=8.5)
        ax.set_xlim(1.6, 2.6)
        ax.set_xticks([1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
        ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[2, 0])
    for i, (lab, r) in enumerate(reps.items()):
        density(ax, r["site"], COLORS[i % len(COLORS)], 1.6, 2.2, fill=True)
    ax.set_xlabel("Mg to nearest\nprotein/GTP oxygen (\u00c5)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.set_xlim(1.6, 2.2)
    ax.set_xticks([1.6, 1.8, 2.0, 2.2])
    ax.tick_params(labelsize=7.5)

    sub = gs[2, 1].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.10)
    axc = fig.add_subplot(sub[0])
    axt = fig.add_subplot(sub[1])
    for i, (lab, r) in enumerate(reps.items()):
        c = COLORS[i % len(COLORS)]
        density(axc, r["cis"], c, 88.0, 92.0, fill=True)
        density(axt, r["trans"], c, 160.0, 180.0, fill=True)
    axc.axvline(90, color="0.6", lw=0.5, ls=":")
    axt.axvline(180, color="0.6", lw=0.5, ls=":")
    axc.set_xlim(88, 92)
    axt.set_xlim(160, 180)
    axc.set_xticks([90])
    axt.set_xticks([170, 180])
    axc.set_ylabel("density", fontsize=8.5)
    axt.set_yticklabels([])
    axc.set_title("cis", fontsize=8)
    axt.set_title("trans", fontsize=8)
    for a in (axc, axt):
        a.tick_params(labelsize=7.5)
    axc.set_xlabel("O\u2013Mg\u2013O (\u00b0)", fontsize=8.5)
    axc.xaxis.set_label_coords(1.15, -0.16)

    fig.legend(handles=[Line2D([], [], color=COLORS[i % len(COLORS)], lw=1.6,
                               label=lab)
                        for i, lab in enumerate(reps)],
               loc="outside lower center", ncol=len(reps), fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_pme_mg_shell.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pme_mg_shell.pdf")


def figure_throughput(logs):
    have = {l: g for l, g in logs.items() if g is not None}
    if not have:
        print("no logs to plot")
        return
    pooled = np.concatenate([g["rate"] for g in have.values()])
    lo = max(0.0, np.percentile(pooled, 0.5) - 15)
    hi = np.percentile(pooled, 99.5) + 15

    fig, ax = plt.subplots(figsize=(COL_W, 2.4), constrained_layout=True)
    for i, (lab, g) in enumerate(have.items()):
        density(ax, g["rate"], COLORS[i % len(COLORS)], lo, hi, fill=True)
    density(ax, pooled, "#333333", lo, hi)
    ax.lines[-1].set_linestyle("--")
    ax.lines[-1].set_linewidth(1.6)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("throughput (ns/day)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    handles = [Line2D([], [], color=COLORS[i % len(COLORS)], lw=1.4, label=lab)
               for i, lab in enumerate(have)]
    handles.append(Line2D([], [], color="#333333", lw=1.6, ls="--",
                          label=f"pooled ({np.median(pooled):.0f})"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_pme_throughput.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pme_throughput.pdf")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not HAVE_KDE:
        print("# scipy not found -- histograms instead of KDE")

    shells, logs = {}, {}
    for d, lab in DIRS.items():
        shells[lab] = rawio.get("kras_pme_mg", lab, lambda d=d: analyse(d))
        logs[lab] = read_log(d)
    print()

    print("=== Mg coordination")
    for lab in shells:
        summarise_shell(lab, shells[lab])

    print("=== throughput")
    for lab in logs:
        summarise_log(lab, logs[lab])

    have = [logs[l] for l in logs if logs[l] is not None]
    if have:
        pooled = np.concatenate([g["rate"] for g in have])
        print(f"pooled over {len(have)} replicates, {len(pooled)} records: "
              f"median {np.median(pooled):.1f} ns/day, "
              f"5th-95th {np.percentile(pooled, 5):.1f} - "
              f"{np.percentile(pooled, 95):.1f}")
        print()

    figure_shell(shells)
    figure_throughput(logs)