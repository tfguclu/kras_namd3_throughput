"""
KRAS Mg2+ octahedral coordination analysis.
Prints ONE summary block at the end; writes full data to mg_coord.npz and,
as text, mg_coord_distances.dat (frame, CN, six Mg-O distances) and
mg_coord_angles.dat (frame, fifteen O-Mg-O angles).
mdtraj 1.9.x compatible.
"""
import numpy as np
import mdtraj as md
from collections import Counter
from itertools import combinations

PSF, DCD = 'step3_input.psf', 'step5_production_1.dcd'
CUT_NM, CHUNK = 0.26, 500
OUT = 'mg_coord'

top = md.load_psf(PSF)
mg  = int(top.select('resname MG and name MG')[0])
pool = top.select('(symbol O or symbol N) and not index %d' % mg)
pairs = np.column_stack([np.full_like(pool, mg), pool])

def lbl(i):
    a = top.atom(int(i))
    return f"{a.residue.name}{a.residue.resSeq}-{a.name}"

# pass 1: occupancy + CN per frame
cn_list, occ = [], Counter()
for ch in md.iterload(DCD, top=PSF, chunk=CHUNK):
    d = md.compute_distances(ch, pairs, periodic=True)
    in_sh = d < CUT_NM
    cn_list.extend(in_sh.sum(axis=1).tolist())
    for f in range(ch.n_frames):
        for a in pool[in_sh[f]]:
            occ[int(a)] += 1

cn = np.array(cn_list)
n  = len(cn)
canon = [a for a, _ in occ.most_common(6)]

# pass 2: distances + angles for the six canonical ligands
cp    = np.column_stack([np.full(6, mg), canon])
trips = np.array([[a, mg, b] for a, b in combinations(canon, 2)])
dists, angs = [], []
for ch in md.iterload(DCD, top=PSF, chunk=CHUNK):
    dists.append(md.compute_distances(ch, cp, periodic=True))
    angs.append(md.compute_angles(ch, trips, periodic=True))
dists = np.concatenate(dists) * 10.0          # nm -> A
angs  = np.degrees(np.concatenate(angs))

labs = [lbl(a) for a in canon]
np.savetxt(f'{OUT}_distances.dat', np.column_stack([np.arange(n), cn, dists]),
           fmt=['%d', '%d'] + ['%.4f'] * dists.shape[1],
           header='frame CN ' + ' '.join(f'd({l})/A' for l in labs))
np.savetxt(f'{OUT}_angles.dat', np.column_stack([np.arange(n), angs]),
           fmt=['%d'] + ['%.2f'] * angs.shape[1],
           header='frame ' + ' '.join(f'{labs[i]}|{labs[j]}' for i, j in combinations(range(6), 2)))
np.savez_compressed(f'{OUT}.npz',
                    cn=cn,
                    labels=np.array([lbl(a) for a in canon]),
                    distances_A=dists,
                    angles_deg=angs)

# === single summary print =================================================
out  = [f"Mg2+ coordination | {n} frames | <CN>={cn.mean():.2f}  CN6={100*(cn==6).mean():.1f}%"]
out += [f"{'ligand':<16}{'<d>/A':>8}{'sd':>7}{'occ%':>8}"]
for a, d in zip(canon, dists.T):
    out += [f"{lbl(a):<16}{d.mean():8.2f}{d.std():7.2f}{100*occ[a]/n:8.1f}"]
cis = angs < 135
out += [f"cis <ang>={angs[cis].mean():.1f}  trans <ang>={angs[~cis].mean():.1f}"]
print("\n".join(out))
