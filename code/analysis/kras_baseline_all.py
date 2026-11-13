"""
Reference (baseline CHARMM-GUI) replicates: Mg shell, Switch I, throughput.

Run from KRAS_unbound_basarim/ :

    python kras_baseline_all.py

For each replicate it
  1. concatenates the chained step5_production*.dcd pieces and computes the
     Switch-I collective variable and the Mg first-shell composition,
  2. reads the matching step5_production*.log files for throughput and
     wall-clock time,
and writes

    fig_baseline_mg_shell.pdf    shell densities, one curve per replicate
    fig_baseline_switch1.pdf     Switch I time evolution
    fig_baseline_throughput.pdf  throughput density, per replicate and pooled

A per-piece wall-clock table is printed. Nothing is written to the replicate
directories.

Residue numbering: PSF resid 0 = leading GLY, so canonical KRAS residue N sits
at resid N. Selected atoms are printed once per replicate -- check them.
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
DIRS = {
    "KRAS_unbound_mg_gtp_1/namd": "rep1",
    "KRAS_unbound_mg_gtp_2/namd": "rep2",
    "KRAS_unbound_mg_gtp_3/namd": "rep3",
    "KRAS_unbound_mg_gtp_4/namd": "rep4",
}
PSF = "step3_input.psf"
DCD_GLOB = "step5_production*.dcd"
LOG_GLOB = "step5_production*.log"

TIMESTEP_FS = 2.0            # reference production timestep
DCDFREQ = 50000
FRAME_PS = TIMESTEP_FS * DCDFREQ / 1000.0      # 100 ps per frame

MIN_DWELL_NS = 5.0
MG_SHELL_CUT = 2.6
TRANS_MIN = 135.0
CLOSED_MAX, OPEN_MIN = 45.0, 90.0
DROP_FIRST = 1               # cold-start timing records per log

LIGANDS = ["SER17-OG", "THR35-OG1", "GTP-O2B", "GTP-O2G"]
COLORS = ["#1f6fb4", "#c8571f", "#3f8f5c", "#7d5ba6"]
COL_W = 3.4

plt.rcParams.update({
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

DWELL = max(1, int(round(MIN_DWELL_NS * 1000.0 / FRAME_PS)))

RE_TIMING = re.compile(
    r"^TIMING:\s+(\d+)\s+CPU:\s+[\d.eE+-]+,.*?Wall:\s+([\d.eE+-]+),"
    r".*?([\d.]+)\s+ns/days", re.M)
RE_PERF = re.compile(r"^PERFORMANCE:\s+(\d+)\s+averaging\s+([\d.]+)\s+ns/day",
                     re.M)
RE_WALLCLOCK = re.compile(r"^WallClock:\s+([\d.eE+-]+)", re.M)


def order(path):
    """Base file first, then _1, _2, ... in numeric order."""
    m = re.search(r"_(\d+)\.(dcd|log)$", os.path.basename(path))
    return int(m.group(1)) if m else -1


# ---- trajectory -------------------------------------------------------------
def analyse(directory):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_angles, calc_bonds, distance_array
    from MDAnalysis.analysis import align

    dcds = sorted(glob.glob(os.path.join(directory, DCD_GLOB)), key=order)
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

    s17 = one("resid 17 and name CA", "Ser17 CA")
    q25 = one("resid 25 and name CA", "Gln25 CA")
    t35 = one("resid 35 and name CA", "Thr35 CA")
    core = one("resid 5 and name CA", "core CA")
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
    sw1 = u.select_atoms("resid 30:40 and name CA")
    site = u.atoms[np.array([a.index for a in named.values()])]

    print(f"# {directory}: {len(dcds)} DCD pieces, {FRAME_PS:.0f} ps per frame")
    print(f"#   CV {s17.resname}{s17.resid} / {q25.resname}{q25.resid} (vertex)"
          f" / {t35.resname}{t35.resid}   ion {mg.resname}{mg.resid}")
    print(f"#   Switch-I RMSD over {sw1.n_atoms} CA, oxygen pool {pool.n_atoms}")

    keys = list(named)
    cv, cn, nwat, site_d, cis, trans, rmsd, tc, dists = ([] for _ in range(9))

    u.trajectory[0]
    ref = sw1.positions.copy()

    for ts in u.trajectory:
        box = ts.dimensions
        cv.append(np.degrees(calc_angles(s17.position, q25.position,
                                         t35.position, box=box)))

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

        mob = sw1.positions.copy()
        _, r = align.rotation_matrix(mob - mob.mean(axis=0),
                                     ref - ref.mean(axis=0))
        rmsd.append(r)
        tc.append(calc_bonds(t35.position, core.position, box=box))

    dists = np.asarray(dists, dtype=np.float32)
    return {
        "t": np.arange(len(cv), dtype=np.float32) * FRAME_PS / 1000.0,
        "cv": np.asarray(cv, dtype=np.float32),
        "cn": np.asarray(cn, dtype=np.int16),
        "nwat": np.asarray(nwat, dtype=np.int16),
        "cis": np.asarray(cis, dtype=np.float32),
        "trans": np.asarray(trans, dtype=np.float32),
        "site": np.asarray(site_d, dtype=np.float32),
        "rmsd": np.asarray(rmsd, dtype=np.float32),
        "tc": np.asarray(tc, dtype=np.float32),
        "dist": {k: dists[:, j] for j, k in enumerate(keys)},
    }


def episodes(mask):
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s.tolist(), (e - s).tolist()))


def finish(r, label):
    r["label"] = label
    t35 = r["dist"]["THR35-OG1"]
    inshell = t35 < MG_SHELL_CUT
    eps = episodes(~inshell)
    real = [(s, L) for s, L in eps if L >= DWELL]
    r["exit"] = real[0][0] if real else None
    r["n_eps"] = len(eps)
    r["n_real"] = len(real)
    r["occ"] = 100 * inshell.mean()
    r["longest"] = max((L for _, L in eps), default=0) * FRAME_PS / 1000.0
    return r


def summarise_traj(r):
    print(f"--- {r['label']}: {len(r['t'])} frames, {r['t'][-1]:.0f} ns")
    print(f"  Thr35 occupancy {r['occ']:.1f}%, {r['n_eps']} cutoff excursion(s),"
          f" longest {r['longest']:.1f} ns")
    if r["exit"] is not None:
        print(f"  first sustained exit {r['t'][r['exit']]:.0f} ns "
              f"({r['n_real']} in total)")
    else:
        print(f"  no excursion >= {MIN_DWELL_NS:.0f} ns "
              f"-- Thr35 coordinated throughout")
    print(f"  CN==6 {100 * np.mean(r['cn'] == 6):.1f}%, waters "
          f"{r['nwat'].mean():.2f}, Mg-to-site {r['site'].mean():.3f} A")
    for k in LIGANDS:
        if k not in r["dist"]:
            continue
        x = r["dist"][k]
        xi = x[x < MG_SHELL_CUT]
        print(f"    {k:<11} {xi.mean():.3f} +/- {xi.std():.3f} A   "
              f"occupancy {100 * np.mean(x < MG_SHELL_CUT):5.1f}%")
    print(f"  cis {np.nanmean(r['cis']):.2f} +/- {np.nanstd(r['cis']):.2f}, "
          f"trans {np.nanmean(r['trans']):.2f} +/- "
          f"{np.nanstd(r['trans']):.2f} deg")
    st = np.where(r["cv"] < CLOSED_MAX, 0, np.where(r["cv"] > OPEN_MIN, 2, 1))
    print(f"  CV mean {r['cv'].mean():.1f} max {r['cv'].max():.1f} deg, "
          f"closed {100 * np.mean(st == 0):.1f}%, "
          f"open {100 * np.mean(st == 2):.1f}%")
    print(f"  Switch-I bb RMSD mean {r['rmsd'].mean():.2f} "
          f"max {r['rmsd'].max():.2f} A")
    print()


# ---- logs -------------------------------------------------------------------
def dhm(sec):
    if sec is None:
        return "--"
    m = int(round(sec / 60.0))
    d, rem = divmod(m, 1440)
    h, mi = divmod(rem, 60)
    return f"{d}:{h:02d}:{mi:02d}"


def read_logs(directory):
    files = sorted(glob.glob(os.path.join(directory, LOG_GLOB)), key=order)
    if not files:
        print(f"#   no log matched {LOG_GLOB} in {directory}")
        return None

    rate, pieces = [], []
    for f in files:
        text = open(f, errors="replace").read()
        hits = RE_TIMING.findall(text)
        if hits:
            rows = [(int(a), float(b), float(c)) for a, b, c in hits]
        else:
            perf = RE_PERF.findall(text)
            if not perf:
                print(f"#   {os.path.basename(f)}: no timing records")
                continue
            rows = [(int(a), np.nan, float(b)) for a, b in perf]
        rows = rows[DROP_FIRST:]
        if not rows:
            continue
        wc = RE_WALLCLOCK.findall(text)
        wall = float(wc[-1]) if wc else rows[-1][1]
        pieces.append({"name": os.path.basename(f), "steps": rows[-1][0],
                       "wall": wall, "exact": bool(wc)})
        rate += [r[2] for r in rows]

    if not rate:
        return None
    return {"rate": np.asarray(rate), "pieces": pieces}


def summarise_logs(label, g):
    if g is None:
        return
    r = g["rate"]
    print(f"--- {label}: {len(g['pieces'])} piece(s), {len(r)} timing records")
    print(f"  ns/day  mean {r.mean():.1f}  median {np.median(r):.1f}  "
          f"sd {r.std():.1f}  min {r.min():.1f}  max {r.max():.1f}")
    print(f"  {'piece':<26}{'steps':>12}{'sim (ns)':>10}"
          f"{'wall d:hh:mm':>14}{'ns/day':>9}")
    tw, tsteps = 0.0, 0
    for p in g["pieces"]:
        sim = p["steps"] * TIMESTEP_FS / 1e6
        eff = sim / (p["wall"] / 86400.0) if p["wall"] else float("nan")
        flag = "" if p["exact"] else "  *"
        print(f"  {p['name']:<26}{p['steps']:>12,}{sim:>10.1f}"
              f"{dhm(p['wall']):>14}{eff:>9.1f}{flag}")
        tw += p["wall"] or 0.0
        tsteps += p["steps"]
    tsim = tsteps * TIMESTEP_FS / 1e6
    print(f"  {'TOTAL':<26}{tsteps:>12,}{tsim:>10.1f}{dhm(tw):>14}"
          f"{tsim / (tw / 86400.0) if tw else float('nan'):>9.1f}")
    print()


# ---- figures ----------------------------------------------------------------
def roll(x, w):
    x = np.asarray(x, dtype=float)
    if w < 3 or x.size < w:
        return x
    xp = np.pad(x, w // 2, mode="edge")
    idx = np.arange(x.size)[:, None] + np.arange(w)[None, :]
    return np.median(xp[idx], axis=1)


def density(ax, x, color, lo, hi, fill=False, lw=1.3, ls="-", label=None):
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


def figure_shell(reps):
    fig = plt.figure(figsize=(COL_W, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    for k, lig in enumerate(LIGANDS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        for i, r in enumerate(reps):
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
    for i, r in enumerate(reps):
        density(ax, r["site"], COLORS[i % len(COLORS)], 1.6, 2.2, fill=True)
    ax.set_xlabel("Mg to nearest\nprotein/GTP oxygen (\u00c5)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.set_xlim(1.6, 2.2)
    ax.set_xticks([1.6, 1.8, 2.0, 2.2])
    ax.tick_params(labelsize=7.5)

    sub = gs[2, 1].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.10)
    axc = fig.add_subplot(sub[0])
    axt = fig.add_subplot(sub[1])
    for i, r in enumerate(reps):
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
                               label=r["label"]) for i, r in enumerate(reps)],
               loc="outside lower center", ncol=len(reps), fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_baseline_mg_shell.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_baseline_mg_shell.pdf")


def figure_switch1(reps):
    n = len(reps)
    fig = plt.figure(figsize=(COL_W, 1.25 * n + 1.7), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1.0] * n + [1.05])
    tmax = max(r["t"][-1] for r in reps)

    for i, r in enumerate(reps):
        c = COLORS[i % len(COLORS)]
        w = max(3, (len(r["t"]) // 300) | 1)
        t35 = r["dist"]["THR35-OG1"]

        ax = fig.add_subplot(gs[i])
        ax.axhspan(0, CLOSED_MAX, color="#eef1f4", lw=0)
        ax.axhspan(OPEN_MIN, 180, color="#e2e6ea", lw=0)
        ax.fill_between(r["t"], 0, 180, where=(t35 >= MG_SHELL_CUT),
                        color="#d8b26a", alpha=0.26, lw=0, step="mid")
        ax.plot(r["t"], r["cv"], color=c, lw=0.2, alpha=0.28)
        ax.plot(r["t"], roll(r["cv"], w), color=c, lw=1.0)
        late = True
        if r["exit"] is not None:
            xe = r["t"][r["exit"]]
            late = xe > 0.5 * tmax
            ax.axvline(xe, color="k", lw=0.8, ls="--")
            ax.text(xe, 172, f"{xe:.0f} ns " if late else f" {xe:.0f} ns",
                    fontsize=7.5, va="top", ha="right" if late else "left")
        ax.set_ylim(0, 180)
        ax.set_xlim(0, tmax)
        ax.set_yticks([0, 90, 180])
        ax.set_ylabel("CV (\u00b0)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.text(0.015 if late else 0.985, 0.92, r["label"],
                transform=ax.transAxes, ha="left" if late else "right",
                va="top", fontsize=8.5, color=c)
        if i == n - 1:
            ax.set_xlabel("time (ns)", fontsize=8.5)
        else:
            ax.set_xticklabels([])

        axr = ax.twinx()
        axr.plot(r["t"], roll(t35, w), color="0.30", lw=0.7)
        axr.axhline(MG_SHELL_CUT, color="0.25", lw=0.5, ls=":")
        axr.set_ylim(0, 30)
        axr.set_yticks([0, 15, 30])
        axr.set_ylabel("Thr35\u2013Mg (\u00c5)", color="0.30", fontsize=8.5)
        axr.tick_params(axis="y", colors="0.30", labelsize=7.5)
        axr.spines["right"].set_visible(True)
        axr.spines["right"].set_color("0.6")

    ax = fig.add_subplot(gs[n])
    ax.axvspan(0, CLOSED_MAX, color="#eef1f4", lw=0, zorder=0)
    ax.axvspan(OPEN_MIN, 180, color="#e2e6ea", lw=0, zorder=0)
    for i, r in enumerate(reps):
        density(ax, r["cv"], COLORS[i % len(COLORS)], 0, 180, label=r["label"])
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Switch I CV (\u00b0)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_baseline_switch1.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_baseline_switch1.pdf")


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
        density(ax, g["rate"], COLORS[i % len(COLORS)], lo, hi, fill=True,
                label=lab)
    density(ax, pooled, "#333333", lo, hi, lw=1.6, ls="--",
            label=f"pooled ({np.median(pooled):.0f})")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("throughput (ns/day)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=8, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_baseline_throughput.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_baseline_throughput.pdf")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not HAVE_KDE:
        print("# scipy not found -- histograms instead of KDE")

    reps, logs = [], {}
    for d, lab in DIRS.items():
        reps.append(finish(rawio.get("kras_baseline_all", lab, lambda d=d: analyse(d)), lab))
        logs[lab] = read_logs(d)
    print()

    print("=== trajectories")
    for r in reps:
        summarise_traj(r)

    print("=== throughput and wall time")
    for lab in logs:
        summarise_logs(lab, logs[lab])

    have = [g for g in logs.values() if g is not None]
    if have:
        pooled = np.concatenate([g["rate"] for g in have])
        print(f"pooled over {len(have)} replicates, {len(pooled)} records: "
              f"median {np.median(pooled):.1f} ns/day, "
              f"5th-95th {np.percentile(pooled, 5):.1f} - "
              f"{np.percentile(pooled, 95):.1f}")
        print("\n  * wall time from the last TIMING record (no WallClock "
              "line); lower bound.\n")

    figure_shell(reps)
    figure_switch1(reps)
    figure_throughput(logs)
