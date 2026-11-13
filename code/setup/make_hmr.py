#!/usr/bin/env python
# HMR by editing ONLY the mass column of the original PSF text.
# Preserves the original header (CMAP, resid 0, XPLOR format) exactly, so NAMD
# reads it identically to step3_input.psf -- only masses change.

PSF_IN  = 'step3_input.psf'
PSF_OUT = 'step3_input_hmr.psf'
H_TARGET = 3.024
WATER = {'TIP3','TIP','TIP3P','HOH','WAT','SPC','SPCE'}
H_LIGHT = 1.5

with open(PSF_IN) as f:
    lines = f.readlines()

# locate NATOM section
natom_i = next(i for i,l in enumerate(lines) if '!NATOM' in l)
natom = int(lines[natom_i].split()[0])
atom_start = natom_i + 1

# XPLOR EXT atom line columns (whitespace-separated):
# idx segid resid resname atomname atomtype charge mass [extra]
# We parse fields, edit mass, and REWRITE preserving original spacing width.
# Original line example:
# "         1 PROA     0        GLY      N        NH3     -0.300000       14.0070           0"

# First pass: read atoms into a list of field-lists, record which are H and their heavy neighbor.
atoms = []
for k in range(natom):
    ln = lines[atom_start + k]
    parts = ln.split()
    # parts: [idx, segid, resid, resname, atomname, atomtype, charge, mass, ...]
    atoms.append({
        'raw': ln,
        'idx': int(parts[0]),
        'segid': parts[1],
        'resid': parts[2],
        'resname': parts[3],
        'name': parts[4],
        'type': parts[5],
        'charge': float(parts[6]),
        'mass': float(parts[7]),
        'rest': parts[8:],
    })

# We need bonds to find each H's heavy partner. Parse the !NBOND section.
# find NBOND
nbond_i = next(i for i,l in enumerate(lines) if '!NBOND' in l)
nbond = int(lines[nbond_i].split()[0])
# bonds are pairs of 1-based atom indices, 4 pairs (8 ints) per line typically
bond_ints = []
j = nbond_i + 1
while len(bond_ints) < nbond*2:
    bond_ints.extend(int(x) for x in lines[j].split())
    j += 1
bonds = [(bond_ints[2*i]-1, bond_ints[2*i+1]-1) for i in range(nbond)]  # 0-based

# build neighbor map (only need heavy partner of each H)
from collections import defaultdict
nbr = defaultdict(list)
for a,b in bonds:
    nbr[a].append(b); nbr[b].append(a)

# apply HMR
n_done = 0
for i,at in enumerate(atoms):
    if at['mass'] >= H_LIGHT:      # not a hydrogen
        continue
    if at['resname'] in WATER:     # skip water
        continue
    # find bonded heavy atom
    heavy = None
    for nb in nbr[i]:
        if atoms[nb]['mass'] >= H_LIGHT:
            heavy = nb; break
    if heavy is None:
        continue
    add = H_TARGET - at['mass']
    at['mass'] = H_TARGET
    atoms[heavy]['mass'] -= add
    n_done += 1

# rewrite atom lines, preserving the original column layout.
# Original mass field is column-aligned; we reformat mass to match width "%14.4f"-ish.
# Safest: rebuild the line with the SAME format CHARMM-GUI uses.
def fmt(at):
    # CHARMM EXT XPLOR format:
    # %10d %-8s %-8s %-8s %-8s %-6s %10.6f %13.4f %11d  (approx, matching original)
    extra = at['rest'][0] if at['rest'] else '0'
    return (f"{at['idx']:10d} {at['segid']:<8s} {at['resid']:<8s} {at['resname']:<8s} "
            f"{at['name']:<8s} {at['type']:<6s} {at['charge']:10.6f}   {at['mass']:12.4f}"
            f"{int(extra):11d}\n")

for k in range(natom):
    lines[atom_start + k] = fmt(atoms[k])

with open(PSF_OUT,'w') as f:
    f.writelines(lines)

print(f"repartitioned {n_done} solute H")
print(f"wrote {PSF_OUT} (header preserved from original)")
# quick verify
print("header:", lines[0].strip())
print("first atom:", lines[atom_start].strip())
print("first H:", lines[atom_start+1].strip())