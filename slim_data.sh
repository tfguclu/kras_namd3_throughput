#!/usr/bin/env bash
# Shrink data/: keep one copy of files that are identical in every run directory
# (system files, force field, equilibration) in data/system/ and replace the
# copies by relative symlinks; drop Colvars restart states (*.colvars.state),
# which no analysis reads.  Run from the repository root:  bash slim_data.sh
set -euo pipefail
cd "$(dirname "$0")/data"
mkdir -p system
find . -name '*.colvars.state' -delete

SHARED=( step3_input.psf step3_input.pdb step3_input.crd step3_input.str
         step3_input_hmr.psf step4_equilibration.inp )
for name in "${SHARED[@]}"; do
  first=$(find . -path ./system -prune -o -type f -name "$name" -print | sort | head -1)
  [ -n "$first" ] || continue
  [ -f "system/$name" ] || cp -p "$first" "system/$name"
  ref=$(md5sum "system/$name" 2>/dev/null | cut -d' ' -f1 || md5 -q "system/$name")
  find . -path ./system -prune -o -type f -name "$name" -print | while read -r f; do
    sum=$(md5sum "$f" 2>/dev/null | cut -d' ' -f1 || md5 -q "$f")
    if [ "$sum" = "$ref" ]; then
      rel=$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[1],os.path.dirname(sys.argv[2])))" "system/$name" "$f")
      rm "$f"; ln -s "$rel" "$f"
    else
      echo "kept (differs from system/$name): $f"
    fi
  done
done

# equilibration logs: one copy per distinct run
for f in $(find . -path ./system -prune -o -type f -name step4_equilibration.log -print | sort); do
  sum=$(md5sum "$f" 2>/dev/null | cut -d' ' -f1 || md5 -q "$f")
  tgt="system/step4_equilibration_${sum:0:8}.log"
  [ -f "$tgt" ] || cp -p "$f" "$tgt"
  rel=$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[1],os.path.dirname(sys.argv[2])))" "$tgt" "$f")
  rm "$f"; ln -s "$rel" "$f"
done
du -sh .
