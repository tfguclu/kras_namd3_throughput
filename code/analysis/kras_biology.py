#!/usr/bin/env python3
"""
Per-residue structural comparison of the three KRAS-GTP-Mg arms:
baseline MD (2 fs), well-tempered metaD (2 fs, biased) and HMR (4 fs).

Run from KRAS_unbound_basarim/:
    python kras_biology.py            # analyse (cached per replicate) + plot
    python kras_biology.py --plot     # plot from cache only

Per residue, per replicate (then mean +- SEM over replicates):
  * secondary structure  STRIDE (must be on PATH); fraction of frames in
                         helix (H, G, I) and strand (E, B)
  * lDDT-Ca              AlphaFold2 definition (alphafold/model/lddt.py):
                         Ca atoms, 15 A inclusion radius taken in the
                         reference, thresholds 0.5/1/2/4 A, self excluded,
                         per-residue normalisation; no stereochemistry term
  * hydrogen bonds       MDAnalysis HydrogenBondAnalysis, scheme of Guclu et
                         al. 2026 (Proteins, doi:10.1002/prot.70182): D-A <= 3.2 A,
                         D-H-A > 150 deg, vicinal waters (within 4 A of protein
                         or GTP, re-selected every frame); mean H-bonds per frame in which
                         the residue takes part, split into solute
                         (protein/GTP) and water partners
  * RSA                  FreeSASA (Lee-Richards, probe 1.4 A, ProtOr radii)
                         on protein + GTP + Mg heavy atoms, per-residue SASA
                         divided by the theoretical maximum of Tien et al.
                         2013 (PLoS ONE 8:e80635)

Writes  bio_cache/<arm>_<rep>.npz  (binary cache) and
        raw/kras_biology/<arm>_<rep>_{residues.dat, stride.dat.gz, lddt.dat.gz,
        rsa.dat.gz, hbonds.dat.gz}  per-frame measures as text; --plot works
        from these alone, without the DCDs (--export-raw converts an old cache)
        bio_per_residue.csv  per residue mean and SD over all frames, per arm
        bio_summary.csv      mean +- SD over residues (protein, P-loop, Switch I/II)
        bio_pearson.csv      Pearson R between arms of per-residue means and SDs
        fig_biology          per-residue means, arm vs arm, error bars = sigma over
                             frames; R (means) and R_sigma (sigmas) in each panel

Notes
  * metaD frames are unweighted (no bias reweighting) and, by default, only
    frames before the CV ~ 0 lock-in are used (kras_metad_summary.py logic).
  * Residue numbering: PSF resid N = KRAS residue N (resid 0 = extra Gly,
    analysed but not plotted).
"""
import argparse, glob, gzip, os, re, subprocess, sys, tempfile
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ---------------- settings ----------------
ARMS = {
    "baseline": dict(psf="step3_input.psf", dcd="step5_production*.dcd", reps={
        "rep1": "KRAS_unbound_mg_gtp_1/namd",
        "rep2": "KRAS_unbound_mg_gtp_2/namd",
        "rep3": "KRAS_unbound_mg_gtp_3/namd",
        "rep4": "KRAS_unbound_mg_gtp_4/namd"}),
    "HMR": dict(psf="step3_input_hmr.psf", dcd="step5_production*.dcd", reps={
        "rep1": "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3",
        "rep2": "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_2",
        "rep3": "KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3_3"}),
    "metaD": dict(psf="step3_input.psf", dcd=None, reps={  # DCDs: step*_metadynamics_1d
        "rep1": "KRAS_unbound_mg_gtp_metaD/namd",
        "rep2": "KRAS_unbound_mg_gtp_metaD_2/namd",
        "rep3": "KRAS_unbound_mg_gtp_metaD_4/namd"}),
}
ARM_ORDER = ["baseline", "metaD", "HMR"]
ARM_COLOR = {"baseline": "#111111", "metaD": "#E00000", "HMR": "#0177C1"}

STRIDE = {"baseline": 10, "metaD": 10, "HMR": 1}   # one analysed frame per ns:
                         # baseline/metaD save every 100 ps, HMR every 1 ns
METAD_PRE_TRAP = False   # True: drop metaD frames after the CV ~ 0 lock-in
REF_PDB = None           # None: step4_equilibration.coor of baseline rep1
                         # (the equilibrated start shared by all arms);
                         # or e.g. "6VJJ.pdb" (chain A, matched by resid)
REF_CHAIN = "A"
NPROC = 4                # replicates analysed in parallel
WATER = "resname TIP3"
SOLUTE = "protein or resname GTP"
VICINAL = f"byres ({WATER} and around 4.0 ({SOLUTE}))"   # whole vicinal waters
HB_DA, HB_ANGLE = 3.2, 150.0
SASA_SEL = "(protein or resname GTP or resname MG) and not name H*"
CACHE = "bio_cache"
REGIONS = {"P-loop": (10, 17), "Switch I": (30, 40), "Switch II": (60, 76)}
PLOT_RES = (1, 167)

MAX_ASA = dict(ALA=129, ARG=274, ASN=195, ASP=193, CYS=167, GLN=225, GLU=223,
               GLY=104, HIS=224, ILE=197, LEU=201, LYS=236, MET=224, PHE=240,
               PRO=159, SER=155, THR=172, TRP=285, TYR=263, VAL=174)
RESNAME = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HIE": "HIS",
           "HID": "HIS", "HIP": "HIS", "CYX": "CYS"}
ATOMNAME = {("ILE", "CD"): "CD1"}                 # CHARMM -> PDB
ELEM_R = {"C": 1.88, "N": 1.64, "O": 1.46, "S": 1.77, "P": 1.80, "M": 1.73}
HELIX, STRAND = set("HGI"), set("EB")
SS_CODES = "HGIEBbTC"

plt.rcParams.update({            # identical to the other kras_*.py figures
    "font.size": 9.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------- lDDT (AlphaFold2, numpy port) ----------------
def lddt_per_residue(pred, true, cutoff=15.0):
    """pred, true: (L, 3). Port of alphafold/model/lddt.py, per_residue=True."""
    dt = np.sqrt(1e-10 + ((true[:, None] - true[None]) ** 2).sum(-1))
    dp = np.sqrt(1e-10 + ((pred[:, None] - pred[None]) ** 2).sum(-1))
    score_mask = (dt < cutoff) * (1.0 - np.eye(len(true)))
    l1 = np.abs(dt - dp)
    s = 0.25 * ((l1 < 0.5).astype(float) + (l1 < 1.0) + (l1 < 2.0) + (l1 < 4.0))
    norm = 1.0 / (1e-10 + score_mask.sum(-1))
    return norm * (1e-10 + (score_mask * s).sum(-1))


# ---------------- helpers ----------------
def order(p):
    nums = re.findall(r"(\d+)", os.path.basename(p))
    return [int(n) for n in nums] if nums else [-1]


def find_inputs(arm, d):
    cfg = ARMS[arm]
    if arm == "metaD":
        logs = glob.glob(os.path.join(d, "step*_metadynamics_1d.log"))
        pref = sorted((l[:-4] for l in logs),
                      key=lambda p: int(re.search(r"step(\d+)_", os.path.basename(p)).group(1)))
        dcds = [p + ".dcd" for p in pref if os.path.isfile(p + ".dcd")]
        if not dcds:
            raise SystemExit(f"no step*_metadynamics_1d.dcd in {d}")
        psf = os.path.join(d, cfg["psf"])
        return psf, dcds, pref
    dcds = sorted(glob.glob(os.path.join(d, cfg["dcd"])), key=order)
    return os.path.join(d, cfg["psf"]), dcds, None


def log_value(log, key):
    with open(log, errors="ignore") as fh:
        for line in fh:
            m = re.search(r"Info: " + key + r"\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"no '{key}' in {log}")


def metad_frame_cut(d, pref, u):
    """Number of leading frames to keep: those before the CV ~ 0 lock-in."""
    import kras_metad_summary as ms
    D = ms.load_rep(d)
    trap = ms.trap_start(D["t"], D["cv"])
    if trap is None:
        return u.trajectory.n_frames
    dt_ns = log_value(pref[0] + ".log", "TIMESTEP") * log_value(pref[0] + ".log", "DCD FREQUENCY") * 1e-6
    return min(u.trajectory.n_frames, int(trap / dt_ns))


def pdb_lines(names, resnames, resids, xyz):
    out = []
    for i, (n, rn, ri, (x, y, z)) in enumerate(zip(names, resnames, resids, xyz), 1):
        nm = n if len(n) == 4 else " " + n
        out.append(f"ATOM  {i % 100000:5d} {nm:<4s} {rn:>3s} A{ri:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {n[0]}")
    return "\n".join(out) + "\nTER\nEND\n"


def stride_codes(pdb_text, nres_index, tmpdir):
    path = os.path.join(tmpdir, "f.pdb")
    with open(path, "w") as fh:
        fh.write(pdb_text)
    r = subprocess.run(["stride", path], capture_output=True, text=True)
    codes = np.full(len(nres_index), "C", dtype="<U1")
    for line in r.stdout.splitlines():
        if line.startswith("ASG"):
            p = line.split()
            ri = int(re.match(r"-?\d+", p[3]).group(0))
            if ri in nres_index:
                codes[nres_index[ri]] = p[5]
    return codes


# ---------------- per replicate ----------------
def analyse(job):
    arm, rep, d, redo_hb = job
    out = os.path.join(CACHE, f"{arm}_{rep}.npz")
    old = None
    if os.path.isfile(out):
        old = dict(np.load(out))
        if "hb_frames" in old and not redo_hb:
            print(f"[{arm} {rep}] cached", flush=True)
            return out
        print(f"[{arm} {rep}] redoing H-bond pass only (pass 1 reused)", flush=True)
    import MDAnalysis as mda
    import freesasa
    from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

    psf, dcds, pref = find_inputs(arm, d)
    if not dcds or not os.path.isfile(psf):
        print(f"[{arm} {rep}] SKIP: missing {psf if not os.path.isfile(psf) else 'DCDs'}", flush=True)
        return None
    u = mda.Universe(psf, dcds)
    n_use = metad_frame_cut(d, pref, u) if (arm == "metaD" and METAD_PRE_TRAP) else u.trajectory.n_frames
    step = STRIDE[arm]
    frames = np.arange(0, n_use, step)
    print(f"[{arm} {rep}] {psf} | {len(dcds)} DCDs | {u.trajectory.n_frames} frames, "
          f"stride {step} -> {len(frames)} frames spanning 0-{frames[-1]}", flush=True)

    prot = u.select_atoms("protein")
    resids = prot.residues.resids
    ridx = {r: i for i, r in enumerate(resids)}
    nres = len(resids)
    resn = [RESNAME.get(r, r) for r in prot.residues.resnames]

    # --- reference for lDDT
    ca = u.select_atoms("protein and name CA")
    ref = reference_ca(ca.resids)

    # --- STRIDE input: protein heavy atoms, PDB naming
    heavy = u.select_atoms("protein and not name H*")
    h_rn = [RESNAME.get(r, r) for r in heavy.resnames]
    h_nm = [ATOMNAME.get((rn, n), n) for rn, n in zip(h_rn, heavy.names)]

    # --- SASA atoms and radii
    sas = u.select_atoms(SASA_SEL)
    cls = freesasa.Classifier()
    radii = []
    for rn, n in zip(sas.resnames, sas.names):
        rn2 = RESNAME.get(rn, rn)
        r = cls.radius(rn2, ATOMNAME.get((rn2, n), n))
        radii.append(r if r > 0 else ELEM_R.get(n[0].upper(), 1.80))
    in_prot = np.isin(sas.indices, prot.indices)
    sas_res = np.array([ridx[r] if p else -1 for r, p in zip(sas.resids, in_prot)])
    max_asa = np.array([MAX_ASA.get(r, np.nan) for r in resn])

    import time
    t0 = time.time()
    def eta(k, n, what):
        dt = time.time() - t0
        rem = dt / max(k, 1) * (n - k)
        print(f"[{arm} {rep}] {what} {k}/{n}  {dt/60:.1f} min elapsed, ~{rem/60:.0f} min left",
              flush=True)

    # --- pass 1: STRIDE, lDDT, RSA (checkpointed)
    part1 = os.path.join(CACHE, f"{arm}_{rep}.part1.npz")
    if old is not None and len(old["frames"]) == len(frames):
        ss, ld, rsa = old["ss"], old["lddt"], old["rsa"]
        np.savez_compressed(part1, ss=ss, lddt=ld, rsa=rsa)
    elif os.path.isfile(part1):
        P = np.load(part1); ss, ld, rsa = P["ss"], P["lddt"], P["rsa"]
        print(f"[{arm} {rep}] pass 1 loaded from {part1}", flush=True)
    else:
        ss = np.empty((len(frames), nres), dtype="<U1")
        ld = np.empty((len(frames), len(ca)))
        rsa = np.empty((len(frames), nres))
        m = sas_res >= 0
        with tempfile.TemporaryDirectory() as tmp:
            for k, ts in enumerate(u.trajectory[frames]):
                ss[k] = stride_codes(pdb_lines(h_nm, h_rn, heavy.resids, heavy.positions), ridx, tmp)
                ld[k] = lddt_per_residue(ca.positions, ref)
                res = freesasa.calcCoord(sas.positions.ravel(), radii)
                area = np.array([res.atomArea(i) for i in range(len(sas))])
                rsa[k] = np.bincount(sas_res[m], weights=area[m], minlength=nres) / max_asa
                if k % 100 == 0:
                    eta(k, len(frames), "SS/lDDT/RSA frame")
        np.savez_compressed(part1, ss=ss, lddt=ld, rsa=rsa)
        print(f"[{arm} {rep}] pass 1 saved -> {part1}", flush=True)

    # --- pass 2: hydrogen bonds, in chunks so progress is visible
    t0 = time.time()
    sel_all = f"({SOLUTE}) or ({WATER})"
    hb = HydrogenBondAnalysis(u, between=[["protein", SOLUTE], ["protein", WATER]],
                              d_a_cutoff=HB_DA, d_h_a_angle_cutoff=HB_ANGLE,
                              update_selections=True)
    near = f"(({SOLUTE}) or ({VICINAL}))"
    # max_mass 3.5: HMR hydrogens carry ~3.02 amu (lightest heavy atom 5.96 amu);
    # the MDAnalysis default (0.9-1.1 amu) would silently drop every solute H
    hb.hydrogens_sel = f"({hb.guess_hydrogens(sel_all, max_mass=3.5)}) and {near}"
    hb.acceptors_sel = f"({hb.guess_acceptors(sel_all)}) and {near}"
    hb_sol = np.zeros(nres); hb_wat = np.zeros(nres)
    hb_frames = np.zeros((len(frames), nres), dtype=np.int16)   # all partners
    fpos = {int(fr): k for k, fr in enumerate(frames)}
    is_prot = np.zeros(u.atoms.n_atoms, bool); is_prot[prot.indices] = True
    is_wat = np.zeros(u.atoms.n_atoms, bool); is_wat[u.select_atoms(WATER).indices] = True
    a_res = np.array([ridx.get(r, -1) for r in u.atoms.resids])
    CH = 100
    for c0 in range(0, len(frames), CH):
        hb.run(frames=frames[c0:c0 + CH])
        for fr, dn, _, ac, _, _ in hb.results.hbonds:   # frame, donor, H, acceptor, d, angle
            k, dn, ac = fpos[int(fr)], int(dn), int(ac)
            for me, other in ((dn, ac), (ac, dn)):
                if is_prot[me]:
                    if is_wat[other]:
                        hb_wat[a_res[me]] += 1
                        hb_frames[k, a_res[me]] += 1
                    elif a_res[me] != a_res[other] or not is_prot[other]:
                        hb_sol[a_res[me]] += 1   # intra-residue bonds not counted
                        hb_frames[k, a_res[me]] += 1
        eta(min(c0 + CH, len(frames)), len(frames), "H-bond frame")
    hb_sol /= len(frames); hb_wat /= len(frames)

    np.savez_compressed(out, resids=resids, resn=np.array(resn), ca_resids=ca.resids,
                        ss=ss, lddt=ld, rsa=rsa, hb_sol=hb_sol, hb_wat=hb_wat,
                        hb_frames=hb_frames,
                        frames=frames)
    os.remove(part1)
    export_raw(arm, rep, dict(np.load(out)))
    print(f"[{arm} {rep}] done -> {out} and {RAW}/", flush=True)
    return out


_REF = {}
def reference_ca(ca_resids):
    """Ca coordinates of the lDDT reference, in the order of ca_resids."""
    key = tuple(ca_resids)
    if key in _REF:
        return _REF[key]
    import MDAnalysis as mda
    if REF_PDB:
        r = mda.Universe(REF_PDB)
        g = r.select_atoms(f"protein and name CA and chainID {REF_CHAIN}")
        pos = {int(i): p for i, p in zip(g.resids, g.positions)}
        missing = [i for i in ca_resids if i not in pos]
        if missing:
            raise SystemExit(f"reference lacks Ca for resids {missing[:10]}...; "
                             "trim the selection or use the simulation start")
        ref = np.array([pos[i] for i in ca_resids])
    else:
        d = ARMS["baseline"]["reps"]["rep1"]
        psf, dcds, _ = find_inputs("baseline", d)
        coor = os.path.join(d, "step4_equilibration.coor")
        if os.path.isfile(coor):
            r = mda.Universe(psf, coor, format="NAMDBIN")
            print(f"lDDT reference: {coor}", flush=True)
        else:
            r = mda.Universe(psf, dcds[0])
            print(f"lDDT reference: frame 0 of {dcds[0]}", flush=True)
        ref = r.select_atoms("protein and name CA").positions.copy()
    _REF[key] = ref
    return ref


# ---------------- aggregation and plot ----------------
RAW = os.path.join("raw", "kras_biology")


def export_raw(arm, rep, D):
    """Per-frame measures as text: one row per frame, one column per residue."""
    os.makedirs(RAW, exist_ok=True)
    stem = os.path.join(RAW, f"{arm}_{rep}")
    with open(stem + "_residues.dat", "w") as fh:
        fh.write("# resid resname\n")
        for r, n in zip(D["resids"], D["resn"]):
            fh.write(f"{int(r)} {n}\n")
    head = "frame " + " ".join(str(int(r)) for r in D["resids"])
    fr = np.asarray(D["frames"]).reshape(-1, 1)
    # STRIDE codes: one string per frame, one character per residue
    with gzip.open(stem + "_stride.dat.gz", "wt") as fh:
        fh.write("# frame codes (one STRIDE character per residue, residue order as in _residues.dat)\n")
        for f_, row in zip(D["frames"], D["ss"]):
            fh.write(f"{int(f_)} {''.join(row)}\n")
    for key, name, fmt in (("lddt", "lddt", "%.3f"), ("rsa", "rsa", "%.3f"),
                           ("hb_frames", "hbonds", "%d")):
        np.savetxt(stem + f"_{name}.dat.gz", np.hstack([fr, D[key]]),
                   fmt=["%d"] + [fmt] * D[key].shape[1], header=head)


def load_raw(stem):
    res = np.loadtxt(stem + "_residues.dat", dtype=str, ndmin=2, skiprows=1)
    ss, frames = [], []
    with gzip.open(stem + "_stride.dat.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f_, codes = line.split()
            frames.append(int(f_)); ss.append(list(codes))
    D = {"resids": res[:, 0].astype(int), "resn": res[:, 1],
         "frames": np.array(frames), "ss": np.array(ss, dtype="<U1")}
    for key, name in (("lddt", "lddt"), ("rsa", "rsa"), ("hb_frames", "hbonds")):
        D[key] = np.loadtxt(stem + f"_{name}.dat.gz", ndmin=2)[:, 1:]
    D["hb_frames"] = D["hb_frames"].astype(np.int16)
    return D


def load_cache():
    """Replicates from bio_cache/*.npz, or else from the text files in raw/."""
    data = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        if f.endswith(".part1.npz"):
            continue
        arm, rep = os.path.basename(f)[:-4].rsplit("_", 1)
        data.setdefault(arm, {})[rep] = dict(np.load(f))
    for f in sorted(glob.glob(os.path.join(RAW, "*_residues.dat"))):
        arm, rep = os.path.basename(f)[:-len("_residues.dat")].rsplit("_", 1)
        if rep not in data.get(arm, {}):
            data.setdefault(arm, {})[rep] = load_raw(f[:-len("_residues.dat")])
    return data


PROPS = [("helix", "helix fraction"), ("strand", "strand fraction"),
         ("coil", "turn/coil fraction"),
         ("lddt", "lDDT-Cα"), ("hbond", "H-bonds per residue"), ("rsa", "RSA")]
PLOT_PROPS = ["coil", "lddt", "hbond", "rsa"]      # helix/strand stay in the CSVs
SS_CLASS = {"helix": set("HGI"), "strand": set("EB"),
            "coil": set("TCb")}            # turn + coil; STRIDE b = isolated bridge


def frame_values(D):
    """Per-frame, per-residue numeric values (frames x residues)."""
    ss = D["ss"]
    v = {k: np.isin(ss, list(c)).astype(float) for k, c in SS_CLASS.items()}
    v["lddt"] = D["lddt"]; v["rsa"] = D["rsa"]; v["hbond"] = D["hb_frames"].astype(float)
    return v


def summarise(data):
    """Pool all frames of all replicates of an arm; per residue mean and SD over frames."""
    prof = {}
    for arm in ARM_ORDER:
        if arm not in data:
            continue
        reps = list(data[arm].values())
        V = [frame_values(D) for D in reps]
        prof[arm] = dict(resids=reps[0]["resids"], resn=reps[0]["resn"], n=len(reps),
                         nframes=sum(len(D["frames"]) for D in reps))
        for k, _ in PROPS:
            a = np.concatenate([v[k] for v in V], 0)
            prof[arm][k] = (np.nanmean(a, 0), np.nanstd(a, 0, ddof=1))
    arms = list(prof)
    keys = [k for k, _ in PROPS]
    base = prof[arms[0]]
    sel = (base["resids"] >= PLOT_RES[0]) & (base["resids"] <= PLOT_RES[1])

    with open("bio_per_residue.csv", "w") as fh:
        fh.write("resid,resname," + ",".join(f"{a}_{k}_{s}" for a in arms for k in keys
                                             for s in ("mean", "sd")) + "\n")
        for i in np.where(sel)[0]:
            vals = [f"{prof[a][k][j][i]:.4f}" for a in arms for k in keys for j in (0, 1)]
            fh.write(f"{base['resids'][i]},{base['resn'][i]}," + ",".join(vals) + "\n")

    # protein-wide and regional mean +- SD (over residues of the per-residue means)
    with open("bio_summary.csv", "w") as fh:
        fh.write("region,property," + ",".join(f"{a}_mean,{a}_sd" for a in arms) + "\n")
        for reg, (lo, hi) in [("all 1-167", PLOT_RES)] + list(REGIONS.items()):
            m = (base["resids"] >= lo) & (base["resids"] <= hi)
            for k in keys:
                fh.write(f"{reg},{k}," + ",".join(
                    f"{np.nanmean(prof[a][k][0][m]):.3f},{np.nanstd(prof[a][k][0][m], ddof=1):.3f}"
                    for a in arms) + "\n")

    # Pearson R between arms over residues 1-167, of the per-residue means
    # (stat 0) and of the per-residue SDs over frames (stat 1)
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    R = {}
    with open("bio_pearson.csv", "w") as fh:
        fh.write("property,statistic," + ",".join(f"{a}_vs_{b}" for a, b in pairs) + "\n")
        for k in keys:
            for st, name in ((0, "mean"), (1, "sd")):
                row = []
                for a, b in pairs:
                    x, y = prof[a][k][st][sel], prof[b][k][st][sel]
                    ok = np.isfinite(x) & np.isfinite(y)
                    r = (np.corrcoef(x[ok], y[ok])[0, 1]
                         if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan)
                    R[(k, st, a, b)] = r
                    row.append(f"{r:.3f}")
                fh.write(f"{k},{name}," + ",".join(row) + "\n")
    print("wrote bio_per_residue.csv, bio_summary.csv, bio_pearson.csv")
    for a in arms:
        print(f"  {a}: {prof[a]['n']} replicates, {prof[a]['nframes']} frames")
    return prof, pairs, R, sel


def plot(prof, pairs, R, sel):
    labels = dict(PROPS)
    # --- one scatter matrix: rows = property, columns = pair of arms.
    #     points = per-residue means; error bars = per-residue sigma over frames;
    #     R = Pearson of the means, R_sigma = Pearson of the sigmas
    PHYS = {"coil": (0, 1), "helix": (0, 1), "strand": (0, 1), "lddt": (0, 1),
            "rsa": (0, None), "hbond": (0, None)}
    fig, axes = plt.subplots(len(PLOT_PROPS), len(pairs),
                             figsize=(2.2 * len(pairs) + 0.6, 2.2 * len(PLOT_PROPS)),
                             constrained_layout=True, squeeze=False)
    for i, k in enumerate(PLOT_PROPS):
        mu = np.concatenate([prof[a][k][0][sel] for a in prof])
        sg = np.concatenate([prof[a][k][1][sel] for a in prof])
        lo, hi = np.nanmin(mu - sg), np.nanmax(mu + sg)
        pl, ph = PHYS.get(k, (None, None))
        lo = max(lo, pl) if pl is not None else lo
        hi = min(hi, ph) if ph is not None else hi
        # limits rounded outwards to whole ticks; labels read 0.x (or integers)
        unit = 0.1 if hi - lo < 3 else 1.0
        lo_t = np.floor(lo / unit + 1e-9) * unit
        hi_t = np.ceil(hi / unit - 1e-9) * unit
        step = next(m * unit for m in (1, 2, 5, 10, 20) if (hi_t - lo_t) / (m * unit) <= 5)
        lo_t = np.floor(lo_t / step + 1e-9) * step
        hi_t = np.ceil(hi_t / step - 1e-9) * step
        ticks = np.arange(lo_t, hi_t + step / 2, step)
        pad = 0.04 * (hi_t - lo_t)
        fmt = FormatStrFormatter("%.1f" if unit < 1 else "%.0f")
        for j, (a, b) in enumerate(pairs):
            ax = axes[i, j]
            ax.plot([lo_t - pad, hi_t + pad], [lo_t - pad, hi_t + pad], color="0.6",
                    lw=0.6, ls="--", zorder=0)
            ax.errorbar(prof[a][k][0][sel], prof[b][k][0][sel],
                        xerr=prof[a][k][1][sel], yerr=prof[b][k][1][sel],
                        fmt="none", ecolor="0.75", elinewidth=0.4, capsize=0, zorder=1)
            ax.scatter(prof[a][k][0][sel], prof[b][k][0][sel], s=9, color="#333333",
                       lw=0, alpha=0.9, zorder=2)
            ax.set_xlim(lo_t - pad, hi_t + pad); ax.set_ylim(lo_t - pad, hi_t + pad)
            ax.set_box_aspect(1)
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
            ax.text(0.04, 0.96, f"$\\it{{R}}$ = {R[(k, 0, a, b)]:.2f}\n"
                                f"$\\it{{R}}_{{\\sigma}}$ = {R[(k, 1, a, b)]:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.8))
            ax.set_xlabel(a, fontsize=10, color=ARM_COLOR[a])
            ax.set_ylabel(b, fontsize=10, color=ARM_COLOR[b])
            ax.tick_params(labelsize=9)
            if j == 0:
                ax.annotate(labels[k], xy=(-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontsize=11,
                            fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_biology.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_biology (pdf|png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="plot from cache only")
    ap.add_argument("--redo-hb", nargs="*", default=[], metavar="ARM",
                    help="recompute only the H-bond pass for these arms")
    ap.add_argument("--export-raw", action="store_true",
                    help="write raw/kras_biology/*.dat(.gz) from bio_cache, then plot")
    a = ap.parse_args()
    if a.export_raw:
        for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
            if not f.endswith(".part1.npz"):
                arm, rep = os.path.basename(f)[:-4].rsplit("_", 1)
                export_raw(arm, rep, dict(np.load(f)))
                print(f"exported {arm} {rep}")
        a.plot = True
    if not a.plot:
        from shutil import which
        if which("stride") is None:
            raise SystemExit("stride not found on PATH")
        os.makedirs(CACHE, exist_ok=True)
        jobs = [(arm, rep, d, arm in a.redo_hb) for arm in ARM_ORDER
                for rep, d in ARMS[arm]["reps"].items()
                if os.path.isdir(d) or print(f"WARNING: {d} not found")]
        with Pool(min(NPROC, len(jobs))) as pool:
            pool.map(analyse, jobs)
    data = load_cache()
    if not data:
        raise SystemExit("nothing in cache")
    plot(*summarise(data))


if __name__ == "__main__":
    main()
