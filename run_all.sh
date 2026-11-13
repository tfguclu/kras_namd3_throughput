#!/usr/bin/env bash
# Rebuild every figure and printed summary from data/, without trajectories.
# The trajectory analyses read their per-frame measures from data/raw/; the
# throughput and metadynamics analyses read the NAMD logs and Colvars traces.
#
#   bash run_all.sh
#
# To recompute the per-frame measures from trajectories instead, place the
# *.dcd files back into data/ and run a script with --recompute.
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO/data"
mkdir -p "$REPO/outputs/figures" "$REPO/outputs/summaries" "$REPO/outputs/tables"

run () {  # run <script> [args] : printed output to outputs/summaries/<script>.txt
  echo "== $1"
  python "$REPO/code/analysis/$1.py" "${@:2}" > "$REPO/outputs/summaries/$1.txt" 2>&1 \
    || echo "   failed, see outputs/summaries/$1.txt"
}

run kras_baseline_all          # Fig. 3; reference Mg shell and Switch I (Table 4, 5)
run kras_cpu_sweep             # Fig. 4
run kras_hmr_benchmark         # Fig. 5, Table 1
run kras_hmr_walltime          # wall-clock per chained job (Table 2)
run kras_pme_mg                # PME order 4 Mg shell (Table 4)
run kras_hmr_switch1           # Figs. 6 and 7, Tables 4 and 5
run kras_metad_summary         # Fig. 8, metadynamics throughput
run kras_biology --plot        # Fig. 9 and outputs/tables/bio_*.csv

mv -f ./*.png ./*.pdf "$REPO/outputs/figures/" 2>/dev/null
mv -f ./bio_*.csv "$REPO/outputs/tables/" 2>/dev/null
echo "figures -> outputs/figures, printed summaries -> outputs/summaries"
