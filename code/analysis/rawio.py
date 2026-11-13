"""Plain-text storage of the per-frame measures produced by the analysis scripts.

The trajectory analyses (kras_baseline_all.py, kras_hmr_switch1.py,
kras_pme_mg.py) reduce each replicate to a few per-frame series: the Switch I
angle, the Mg2+ coordination number, the four Mg-O distances, the octahedral
angle means, and so on. These are written once to raw/<script>/<label>.dat, a
whitespace-separated table with a header line naming the columns, so that every
figure and summary can be regenerated from the text files without the DCDs.

    get(script, label, compute)   load raw/<script>/<label>.dat if present,
                                  otherwise call compute(), save, and return it
    save(path, rec) / load(path)  write / read one record

A record is a dict of equal-length 1-D arrays; a nested dict (e.g. the per-ligand
distances) is stored as columns "dist:KEY"; scalars go into the header.
"""
import os
import sys
import numpy as np

RAW_DIR = "raw"
INT_KEYS = {"cn", "nwat", "coordination_number", "n_water_in_shell"}
FORCE = "--recompute" in sys.argv


def _flatten(rec):
    cols, scalars = {}, {}
    for k, v in rec.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                cols[f"{k}:{kk}"] = np.asarray(vv)
        elif np.ndim(v) == 0:
            scalars[k] = v
        else:
            cols[k] = np.asarray(v)
    return cols, scalars


def save(path, rec):
    cols, scalars = _flatten(rec)
    names = list(cols)
    n = {len(c) for c in cols.values()}
    if len(n) != 1:
        raise ValueError(f"columns of unequal length in {path}: {n}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    head = [f"{k} = {v}" for k, v in scalars.items()]
    head.append(" ".join(names))
    data = np.column_stack([cols[k].astype(float) for k in names])
    fmt = ["%d" if k.split(":")[0] in INT_KEYS else "%.6g" for k in names]
    np.savetxt(path, data, fmt=fmt, header="\n".join(head), comments="# ")


def load(path):
    scalars, names = {}, None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line[2:].strip()
            if " = " in body:
                k, v = body.split(" = ", 1)
                try:
                    scalars[k] = int(v)
                except ValueError:
                    try:
                        scalars[k] = float(v)
                    except ValueError:
                        scalars[k] = v
            else:
                names = body.split()
    data = np.loadtxt(path, ndmin=2)
    rec = dict(scalars)
    for j, k in enumerate(names):
        col = data[:, j]
        col = col.astype(np.int16) if k.split(":")[0] in INT_KEYS else col.astype(np.float32)
        if ":" in k:
            top, sub = k.split(":", 1)
            rec.setdefault(top, {})[sub] = col
        else:
            rec[k] = col
    return rec


def get(script, label, compute):
    path = os.path.join(RAW_DIR, script, f"{label}.dat")
    if os.path.isfile(path) and not FORCE:
        print(f"# {label}: per-frame data read from {path}")
        return load(path)
    rec = compute()
    save(path, rec)
    print(f"# {label}: per-frame data written to {path}")
    return rec
