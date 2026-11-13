#!/bin/bash
#SBATCH -q acc_ehpc
#SBATCH -A <your_account>
#SBATCH -J KRAS_wt
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
##SBATCH --mail-type=ALL
##SBATCH --mail-user=username@mail.com
cd $(pwd)
module purge
module load openmpi/4.1.5-gcc-mt cuda fftw/3.3.10-gcc-ompi tcl namd/3.0.1-gcc-smp-cuda

echo "Job started at $(date)"

# Stage 0: minimize + restrained NVT. GPUresident OFF (restraints/NVT).
#namd3 +p20 +setcpuaffinity +devices 0 step4_equilibration.inp > step4_equilibration.log

# Stage 1: 100 ns unbiased NPT production. GPUresident ON.
#namd3 +p20 +setcpuaffinity +devices 0 --GPUresident on step5_production.inp > step5_production.log

# Stage 2: 900 ns metadynamics. GPUresident ON.
namd3 +p20 +setcpuaffinity +devices 0 --GPUresident on step12_2200ns_metadynamics_1d.inp  > step12_2200ns_metadynamics_1d.log

wait
echo "Job ended at $(date)"
