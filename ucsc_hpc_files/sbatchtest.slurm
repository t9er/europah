#!/bin/bash
#SBATCH --job-name=titan0.test   # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=tyyuen@ucsc.edu   # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=00:05:00               # Time limit hrs:min:sec (you may not want this)
#SBATCH --output=serial_test_%j.out   # Standard output and error log
#SBATCH --mem=250M                    # Allocated 250 megabytes of memory for the job.
 
module load python/2.7     
 
python titan0.py
