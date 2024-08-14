#!/bin/bash
#SBATCH --job-name=europaaaa      # Job name
#SBATCH --partition=cpuq             # queue for job submission
#SBATCH --account=cpuq               # queue for job submission
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=tyyuen@ucsc.edu   # Where to send mail
#SBATCH --ntasks=10                # Number of MPI ranks
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=10        # How many tasks on each node
#SBATCH --time=05:00:00              # Time limit hrs:min:sec
#SBATCH --output=test_%j.log     # Standard output and error log

pwd; hostname; date

echo "Running program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS total tasks, with each node getting $SLURM_NTASKS_PER_NODE running on cores."

#module load intel/impi

#mpirun -n 10 --ppn 10 python ./test.py

module load openmpi

mpirun -N 4 --map-by ppr:10:node python ./europa_energytime_mpi.py
date
