#!/usr/bin/env bash
# Copy the NAMD inputs and the small NAMD outputs (logs, Colvars traces) from the
# working directory into data/ of this repository, keeping the original layout.
# Trajectories (*.dcd), restart and velocity files are NOT copied.
#
# Usage:  cd /path/to/KRAS_unbound_basarim
#         bash /path/to/repo/collect_data.sh /path/to/repo
set -euo pipefail
REPO=$(cd "${1:?usage: collect_data.sh /path/to/repo}" && pwd)
DEST="$REPO/data"
mkdir -p "$DEST"

DIRS=(
  KRAS_unbound_mg_gtp_1/namd KRAS_unbound_mg_gtp_2/namd
  KRAS_unbound_mg_gtp_3/namd KRAS_unbound_mg_gtp_4/namd
  KRAS_unbound_mg_gtp_metaD/namd KRAS_unbound_mg_gtp_metaD_2/namd
  KRAS_unbound_mg_gtp_metaD_4/namd
)
# benchmark arms: baseline/pairlist sweeps, PME order 4, HMR replicates
for d in KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_*; do
  [ -d "$d" ] && DIRS+=("$d")
done

# inputs + small outputs, by pattern
INCLUDE=( '*.inp' '*.str' '*.psf' '*.pdb' '*.crd' '*.in' 'sysinfo.dat' 'README'
          '*.log' '*.colvars.traj' '*.colvars.state' '*.pmf' '*.slurm' )
# the common equilibrated starting point (all arms restart from it)
START=( step4_equilibration.coor step4_equilibration.vel step4_equilibration.xsc )

for d in "${DIRS[@]}"; do
  [ -d "$d" ] || { echo "skip (missing): $d"; continue; }
  mkdir -p "$DEST/$d"
  for pat in "${INCLUDE[@]}"; do
    find "$d" -maxdepth 1 -type f -name "$pat" ! -name '*.BAK' ! -name '*.old' \
      -exec cp -p {} "$DEST/$d/" \;
  done
  [ -d "$d/restraints" ] && cp -rp "$d/restraints" "$DEST/$d/"
done
for f in "${START[@]}"; do
  cp -p "KRAS_unbound_mg_gtp_1/namd/$f" "$DEST/KRAS_unbound_mg_gtp_1/namd/" 2>/dev/null || true
done

# CHARMM force-field files: one copy, linked from every run directory
if [ -d KRAS_unbound_mg_gtp_1/namd/toppar ]; then
  rm -rf "$DEST/toppar"; cp -rp KRAS_unbound_mg_gtp_1/namd/toppar "$DEST/toppar"
  for d in "${DIRS[@]}"; do
    [ -d "$DEST/$d" ] || continue
    rel=$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[1],sys.argv[2]))" "$DEST/toppar" "$DEST/$d")
    ln -sfn "$rel" "$DEST/$d/toppar"
    if [ -d "$d/toppar" ] && ! diff -rq "$d/toppar" KRAS_unbound_mg_gtp_1/namd/toppar >/dev/null; then
      echo "WARNING: $d/toppar differs from the shared copy"
    fi
  done
fi

# per-frame measures written by the analysis scripts (raw/), so that every
# figure can be rebuilt without the trajectories
if [ -d raw ]; then
  rm -rf "$DEST/raw"; cp -rp raw "$DEST/raw"
else
  echo "NOTE: no raw/ here yet; run the trajectory analyses once first (see README)"
fi

# remove account names from copied Slurm scripts
find "$DEST" -name '*.slurm' -exec sed -i.bak 's/^#SBATCH -A .*/#SBATCH -A <your_account>/' {} \; \
  -exec rm -f {}.bak \;

# one copy of the files that are identical in every run directory
bash "$REPO/slim_data.sh"

echo; echo "Copied into $DEST: $(du -sh "$DEST" | cut -f1)"
echo "Files above 50 MB (GitHub rejects files over 100 MB):"
find "$DEST" -type f -size +50M -exec ls -lh {} \; || true
