#!/bin/bash
#SBATCH --job-name=gsi_radiance_plots      # Name of the job
#SBATCH --output=plot_gsi%j.log     # Output log file with job ID
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=48:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Adam.Schneider@noaa.gov  # Email address for notifications

#micromamba activate darr_score_sqlalchemy1_env

# Run the MPI Python script using srun
mpirun python plot_gsi_radiance_fit_to_obs.py /media/darr/results/figures/brightness_temperature_error_timeseries2

