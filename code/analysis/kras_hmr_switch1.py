"""
KRAS Switch-I / Mg(2+) analysis and figures for the HMR replicates.

Run from KRAS_unbound_basarim/ (paths below point into KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/):

    python kras_hmr_switch1.py

For each replicate it concatenates the chained DCDs, computes the Switch-I
collective variable and the Mg first-shell composition, prints a summary, and
then writes

    fig_mg_shell.pdf   densities showing the ion stays in place
    fig_switch1.pdf    time evolution of Switch I opening

Nothing is written to the replicate directories; only the two PDFs are saved,
in the working directory.

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
DIRS = {"KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3": "rep1", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_2": "rep2", "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_3": "rep3"}
PSF = "step3_input_hmr.psf"
DCD_GLOB = "step5_production*.dcd"

TIMESTEP_FS = 4.0
DCDFREQ = 250000
FRAME_PS = TIMESTEP_FS * DCDFREQ / 1000.0        # 1000 ps per frame

MIN_DWELL_NS = 5.0        # time outside the shell before an exit counts
MG_SHELL_CUT = 2.6        # A
TRANS_MIN = 135.0         # O-Mg-O angles above this are trans
CLOSED_MAX, OPEN_MIN = 45.0, 90.0    # CV state thresholds, degrees

LIGANDS = ["SER17-OG", "THR35-OG1", "GTP-O2B", "GTP-O2G"]
LIG_LABEL = {"SER17-OG": "Ser17-OG", "THR35-OG1": "Thr35-OG1",
             "GTP-O2B": "GTP O2B", "GTP-O2G": "GTP O2G"}
REF_COLOR = "0.35"
REF_RAW = os.path.join("raw", "kras_baseline_all", "rep{}.dat")   # reference arm
COLORS = ["#1f6fb4", "#c8571f", "#3f8f5c"]

COL_W = 3.4          # IEEE single-column width, inches

plt.rcParams.update({
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

DWELL = max(1, int(round(MIN_DWELL_NS * 1000.0 / FRAME_PS)))


# ---- analysis ---------------------------------------------------------------
def analyse(directory):
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import calc_angles, calc_bonds, distance_array
    from MDAnalysis.analysis import align

    files = glob.glob(os.path.join(directory, DCD_GLOB))
    if not files:
        raise SystemExit(f"no DCD matched {DCD_GLOB} in {directory}")

    def order(p):
        m = re.search(r"_(\d+)\.dcd$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    dcds = sorted(files, key=order)
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

    print(f"# {directory}: {len(dcds)} pieces, {FRAME_PS:.0f} ps per frame")
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
        "time_ns": np.arange(len(cv), dtype=np.float32) * FRAME_PS / 1000.0,
        "cv_deg": np.asarray(cv, dtype=np.float32),
        "coordination_number": np.asarray(cn, dtype=np.int16),
        "ligand_dist_A": {k: dists[:, j] for j, k in enumerate(keys)},
        "n_water_in_shell": np.asarray(nwat, dtype=np.int16),
        "cis_mean_deg": np.asarray(cis, dtype=np.float32),
        "trans_mean_deg": np.asarray(trans, dtype=np.float32),
        "mg_to_site_A": np.asarray(site_d, dtype=np.float32),
        "switch1_bb_rmsd_A": np.asarray(rmsd, dtype=np.float32),
        "thr35_core_dist_A": np.asarray(tc, dtype=np.float32),
    }


# ---- load + summarise -------------------------------------------------------
def episodes(mask):
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s.tolist(), (e - s).tolist()))


def record(raw, label):
    r = {"label": label,
         "t": raw["time_ns"], "cv": raw["cv_deg"],
         "cn": raw["coordination_number"],
         "cis": raw["cis_mean_deg"], "trans": raw["trans_mean_deg"],
         "site": raw["mg_to_site_A"], "nwat": raw["n_water_in_shell"],
         "rmsd": raw["switch1_bb_rmsd_A"], "dist": raw["ligand_dist_A"]}

    t35 = r["dist"]["THR35-OG1"]
    inshell = t35 < MG_SHELL_CUT
    eps = episodes(~inshell)
    real = [(st, L) for st, L in eps if L >= DWELL]
    r["exit"] = real[0][0] if real else None
    r["committed"] = bool(real and real[-1][0] + real[-1][1] >= len(t35))
    r["n_real"] = len(real)
    r["occ"] = 100 * inshell.mean()
    r["longest"] = max((L for _, L in eps), default=0) * FRAME_PS / 1000.0
    return r


def summarise(r):
    print(f"--- {r['label']}: {len(r['t'])} frames, {r['t'][-1]:.0f} ns")
    print(f"  Thr35 occupancy {r['occ']:.1f}%, longest excursion "
          f"{r['longest']:.0f} ns")
    if r["exit"] is not None:
        print(f"  first sustained exit {r['t'][r['exit']]:.0f} ns; "
              f"{r['n_real']} in total; "
              f"{'no return' if r['committed'] else 'returns'}")
    else:
        print(f"  no excursion >= {MIN_DWELL_NS:.0f} ns")
    print(f"  CN==6 {100 * np.mean(r['cn'] == 6):.1f}%, waters "
          f"{r['nwat'].mean():.2f}, Mg-to-site {r['site'].mean():.2f} A")
    for k in LIGANDS:
        x = r["dist"][k]
        xi = x[x < MG_SHELL_CUT]
        print(f"    {k:<11} {xi.mean():.3f} +/- {xi.std():.3f} A   "
              f"occupancy {100 * np.mean(x < MG_SHELL_CUT):5.1f}%")
    print(f"  cis {np.nanmean(r['cis']):.2f}, trans {np.nanmean(r['trans']):.2f}"
          f" deg")
    st = np.where(r["cv"] < CLOSED_MAX, 0, np.where(r["cv"] > OPEN_MIN, 2, 1))
    print(f"  CV mean {r['cv'].mean():.1f} max {r['cv'].max():.1f} deg, "
          f"closed {100 * np.mean(st == 0):.1f}%, open {100 * np.mean(st == 2):.1f}%")
    print()


# ---- figures ----------------------------------------------------------------
def roll(x, w):
    x = np.asarray(x, dtype=float)
    if w < 3 or x.size < w:
        return x
    xp = np.pad(x, w // 2, mode="edge")
    idx = np.arange(x.size)[:, None] + np.arange(w)[None, :]
    return np.median(xp[idx], axis=1)


def density(ax, x, color, lo=None, hi=None, fill=False, label=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    lo = np.nanmin(x) if lo is None else lo
    hi = np.nanmax(x) if hi is None else hi
    if hi <= lo:
        hi = lo + 1e-6
    grid = np.linspace(lo, hi, 400)
    if HAVE_KDE and x.size > 5 and np.ptp(x) > 0:
        y = gaussian_kde(x)(grid)
    else:
        y, edges = np.histogram(x, bins=60, range=(lo, hi), density=True)
        grid = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(grid, y, color=color, lw=1.2, label=label)
    if fill:
        ax.fill_between(grid, y, color=color, alpha=0.15, lw=0)


def load_reference():
    """Pooled per-frame shell data of the four reference replicates (2 fs)."""
    out = None
    for k in range(1, 5):
        path = REF_RAW.format(k)
        if not os.path.isfile(path):
            continue
        r = rawio.load(path)
        if out is None:
            out = {"dist": {l: [] for l in LIGANDS}, "site": [], "cis": [], "trans": []}
        for l in LIGANDS:
            out["dist"][l].append(r["dist"][l])
        for key in ("site", "cis", "trans"):
            out[key].append(r[key])
    if out is None:
        print("# reference raw data not found; Fig. 6 drawn without it")
        return None
    out["dist"] = {l: np.concatenate(v) for l, v in out["dist"].items()}
    for key in ("site", "cis", "trans"):
        out[key] = np.concatenate(out[key])
    return out


def ref_density(ax, x, lo, hi):
    density(ax, x, REF_COLOR, lo=lo, hi=hi)
    ax.lines[-1].set_linestyle("--")


def figure_shell(reps, ref=None):
    fig = plt.figure(figsize=(COL_W, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    for k, lig in enumerate(LIGANDS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        for i, r in enumerate(reps):
            x = r["dist"][lig]
            density(ax, x[x < MG_SHELL_CUT], COLORS[i], lo=1.6, hi=2.6,
                    fill=True)
        if ref is not None:
            x = ref["dist"][lig]
            ref_density(ax, x[x < MG_SHELL_CUT], 1.6, 2.6)
        ax.set_xlabel(f"Mg\u2013O, {LIG_LABEL[lig]} (\u00c5)", fontsize=8.5)
        ax.set_ylabel("density", fontsize=8.5)
        ax.set_xlim(1.6, 2.6)
        ax.set_xticks([1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
        ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(reps):
        density(ax, r["site"], COLORS[i], lo=1.6, hi=2.2, fill=True)
    if ref is not None:
        ref_density(ax, ref["site"], 1.6, 2.2)
    ax.set_xlabel("Mg to nearest\nprotein/GTP oxygen (\u00c5)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.set_xlim(1.6, 2.2)
    ax.set_xticks([1.6, 1.8, 2.0, 2.2])
    ax.tick_params(labelsize=7.5)

    sub = gs[2, 1].subgridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.10)
    axc = fig.add_subplot(sub[0])
    axt = fig.add_subplot(sub[1])
    for i, r in enumerate(reps):
        density(axc, r["cis"], COLORS[i], lo=88.0, hi=92.0, fill=True)
        density(axt, r["trans"], COLORS[i], lo=160.0, hi=180.0, fill=True)
    if ref is not None:
        ref_density(axc, ref["cis"], 88.0, 92.0)
        ref_density(axt, ref["trans"], 160.0, 180.0)
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

    handles = [Line2D([], [], color=COLORS[i], lw=1.6, label=f"HMR {r['label']}")
               for i, r in enumerate(reps)]
    if ref is not None:
        handles.append(Line2D([], [], color=REF_COLOR, lw=1.2, ls="--",
                              label="reference"))
    fig.legend(handles=handles, loc="outside lower center",
               ncol=len(handles), fontsize=8, handlelength=1.6, columnspacing=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_mg_shell.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_mg_shell.pdf")


def figure_switch1(reps):
    n = len(reps)
    fig = plt.figure(figsize=(COL_W, 1.35 * n + 1.7), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1.0] * n + [1.05])
    tmax = max(r["t"][-1] for r in reps)

    for i, r in enumerate(reps):
        c = COLORS[i]
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
            late = xe > 0.5 * tmax           # annotate on the roomier side
            ax.axvline(xe, color="k", lw=0.8, ls="--")
            ax.text(xe, 172, f"{xe:.0f} ns " if late else f" {xe:.0f} ns",
                    fontsize=7.5, va="top", ha="right" if late else "left")
        ax.set_ylim(0, 180)
        ax.set_xlim(0, tmax)
        ax.set_yticks([0, 90, 180])
        ax.set_ylabel("CV (\u00b0)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        # replicate label goes on the side the exit annotation does not use
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
        density(ax, r["cv"], COLORS[i], lo=0, hi=180, label=r["label"])
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Switch I CV (\u00b0)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=8, ncol=1, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_switch1.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_switch1.pdf")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if not HAVE_KDE:
        print("# scipy not found -- histograms instead of KDE")

    reps = [record(rawio.get("kras_hmr_switch1", lab, lambda d=d: analyse(d)), lab)
            for d, lab in DIRS.items()]

    print()
    for r in reps:
        summarise(r)

    figure_shell(reps, load_reference())
    figure_switch1(reps)