# Throughput optimization of KRAS molecular dynamics on MareNostrum 5

Inputs, analysis code and outputs for

> T. F. Guclu, *Throughput optimization of oncogenic KRAS molecular dynamics
> simulations on a GPU supercomputer*, Concurrency and Computation: Practice and
> Experience (submitted).

A 46,326-atom KRAS-GTP-Mg²⁺ system is run in NAMD 3.0.1 (GPU-resident) on one
H100 GPU of MareNostrum 5. Three cumulative changes to the CHARMM-GUI
configuration (removing the explicit pairlist settings, PME interpolation
order 6 → 4, hydrogen mass repartitioning with a 4 fs timestep) raise
throughput from 147 to 494 ns/day. The Mg²⁺ first coordination shell is used as
the acceptance test. Well-tempered metadynamics and per-residue structural
descriptors are compared across the arms.

Trajectories (`*.dcd`) are not included because of their size. Instead, every
quantity measured from them is stored per frame as plain text in `data/raw/`,
so all figures and tables can be rebuilt from this repository alone.
The figures themselves are not distributed here; they appear in the published
article and are regenerated locally by `run_all.sh`.

## Layout

```
code/
  analysis/   analysis scripts (one per figure or table, see below)
  setup/      make_hmr.py: hydrogen mass repartitioning of the CHARMM PSF
  slurm/      submit_mn5.slurm: MareNostrum 5 job script (account removed)
  vmd/        VMD loaders for the reference and HMR trajectories
data/         NAMD inputs and small NAMD outputs, in the original directory layout
  system/                                      shared system files (step3_input.*, HMR PSF,
                                               equilibration input and logs), linked from the run directories
  toppar/                                      CHARMM36m / CGenFF parameters (shared)
  KRAS_unbound_mg_gtp_{1..4}/namd/             reference arm, 4 x 1.64 us, 2 fs
  KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/
    namd_0, namd_1                             core-count sweeps (baseline, pairlist)
    namd_2, namd_2_2, namd_2_3                 PME order 4, 3 x 100 ns
    namd_3, namd_3_2, namd_3_3                 HMR, 4 fs, 3 x 10 us
  KRAS_unbound_mg_gtp_metaD{,_2,_4}/namd/      well-tempered metadynamics, 3 x 1.6 us
  raw/                                         per-frame measures from the trajectories
outputs/
  figures/    created by run_all.sh (not tracked)
  tables/     per-residue descriptors (CSV)
  summaries/  printed output of each analysis script
collect_data.sh   copies inputs and logs from the working directory into data/
slim_data.sh      stores files shared by all run directories once (called by collect_data.sh)
run_all.sh        re-runs the analyses and fills outputs/
```

Each run directory holds the NAMD configuration files (`*.inp`), the CHARMM-GUI
system files (`step3_input.psf/pdb/crd/str`, `restraints/`), the NAMD logs and,
for metadynamics, the Colvars input and `*.colvars.traj` files. Files that are
identical in every run directory are stored once in `data/system/` and linked.
The reference, PME and HMR arms start from the same equilibrated state,
`KRAS_unbound_mg_gtp_1/namd/step4_equilibration.{coor,vel,xsc}`; the
metadynamics arm was equilibrated separately with the same protocol, followed
by a 100 ns unbiased run (`step5_production`) before the biased segments.

## Figures and the scripts that make them

Figure files are written to `outputs/figures/` when the scripts are run.

| Paper | File | Script | Reads |
|---|---|---|---|
| Fig. 3 | `fig_baseline_throughput.png` | `kras_baseline_all.py` | logs + `raw/kras_baseline_all/` |
| Fig. 4 | `fig_cpu_sweep.png` | `kras_cpu_sweep.py` | logs |
| Fig. 5, Table 1 | `fig_throughput.png` | `kras_hmr_benchmark.py` | logs |
| Table 2 | wall-clock per chained job | `kras_hmr_walltime.py` | logs |
| Table 4 (PME 4) | PME order 4 Mg shell | `kras_pme_mg.py` | logs + `raw/kras_pme_mg/` |
| Figs. 6, 7; Tables 4, 5 | `fig_mg_shell.png`, `fig_switch1.png` | `kras_hmr_switch1.py` | `raw/kras_hmr_switch1/` |
| Fig. 8 | `fig_metad.png` | `kras_metad_summary.py` | logs + `*.colvars.traj` |
| Fig. 9 | `fig_biology.png`, `outputs/tables/bio_*.csv` | `kras_biology.py --plot` | `raw/kras_biology/` |
| – | Mg²⁺ octahedral geometry | `mg_coord.py` | trajectory (writes `mg_coord_*.dat`) |

### Per-frame data in `data/raw/`

| Directory | One file per | Columns |
|---|---|---|
| `kras_baseline_all/`, `kras_hmr_switch1/` | replicate (`rep1.dat` …) | time, Switch I angle, coordination number, Mg–O distances to Ser17-OG, Thr35-OG1, GTP O2B and O2G, waters in the shell, mean cis and trans O–Mg–O angles, Mg to nearest protein/GTP oxygen, Switch I backbone RMSD, Thr35–core distance |
| `kras_pme_mg/` | replicate | coordination number, Mg–O distances, waters in the shell, cis/trans angle means, Mg to nearest protein/GTP oxygen |
| `kras_biology/` | arm and replicate | `*_stride.dat.gz` STRIDE code per residue; `*_lddt.dat.gz`, `*_rsa.dat.gz`, `*_hbonds.dat.gz`: frame × residue matrices; `*_residues.dat`: residue order |

Column names are in the header line of each file. Frames are 1 ns apart for the
per-residue data, and 100 ps (reference, PME 4) or 1 ns (HMR) for the Mg²⁺ and
Switch I data.

## Reproducing

Rebuild all figures and tables from the repository (no trajectories needed):

```bash
bash run_all.sh
```

How `data/` was produced, in the original working directory that holds the
trajectories:

```bash
cd /path/to/KRAS_unbound_basarim
# scripts from code/analysis/ (with rawio.py) copied next to the data
python kras_baseline_all.py          # writes raw/kras_baseline_all/
python kras_pme_mg.py                # writes raw/kras_pme_mg/
python kras_hmr_switch1.py           # writes raw/kras_hmr_switch1/
python kras_biology.py --export-raw  # bio_cache/ -> raw/kras_biology/ (or run without the flag to compute)
bash /path/to/this/repo/collect_data.sh /path/to/this/repo   # inputs, logs, Colvars, raw/ -> data/
```

Each trajectory script reads `raw/<script>/` when it exists and the DCD files
otherwise; `--recompute` forces the trajectory analysis.

### Requirements

- NAMD 3.0.1 (SMP CUDA build), for the simulations
- Python 3.10+, NumPy, SciPy, Matplotlib, MDAnalysis (≥ 2.6), mdtraj (for `mg_coord.py`)
- FreeSASA (Python bindings) and STRIDE on `PATH`, for `kras_biology.py`

```bash
pip install numpy scipy matplotlib MDAnalysis mdtraj freesasa
```

### Hydrogen mass repartitioning

```bash
cd data/KRAS_unbound_benchmark_phys/KRAS_unbound_mg_gtp_1/namd_3
python ../../../../code/setup/make_hmr.py   # step3_input.psf -> step3_input_hmr.psf
```

Solute hydrogens are set to 3.024 amu, mass is taken from the bonded heavy atom,
and water is left unchanged. Only the mass column is rewritten, so the CMAP
section, atom types and residue numbering of the CHARMM-GUI PSF are kept.
Analysis tools that identify hydrogens by mass must allow ~3 amu
(`kras_biology.py` uses an upper limit of 3.5 amu).

### Running on MareNostrum 5

`code/slurm/submit_mn5.slurm` requests one GPU and its 20-core share on the
accelerated partition for the 72 h limit of the EuroHPC queue and launches

```bash
namd3 +p20 +setcpuaffinity +devices 0 --GPUresident on <input>.inp > <input>.log
```

Set your own account (`-A`) and queue (`-q`). Restrained equilibration was run
with GPU-resident mode off. Longer trajectories are chains of such jobs, each
restarting from the previous job's restart files.

## Citation

Please cite the article above (citation details will be added on publication).

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0. The figures of the article are not
included and are covered by the article's own license.
