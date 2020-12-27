#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)

#SBATCH --time=2-00:00                         # Runtime in D-HH:MM format
#SBATCH --partition=standard                          # Partition to run in
#SBATCH --mem=1000                          # Memory total in MB (for all cores)
#SBATCH -o out_file/%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e err_file/%j.err                 # File to which STDERR will be written, including job ID

# Execute commands
python3  gibbs_simu.py

# arg1 = seed



