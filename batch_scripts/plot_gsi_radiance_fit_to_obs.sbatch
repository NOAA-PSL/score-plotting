#!/bin/bash
#SBATCH --job-name=gsi_radiance_plots      # Name of the job
#SBATCH --output=plot_gsi%j.log     # Output log file with job ID
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=12:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=bigmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Chesley.Mccoll@noaa.gov  # Email address for notifications

source $HOME/.bash_profile
micromamba activate darr_score_sqlalchemy1_dev_env
#which mpirun
# Run the MPI Python script using mpiexec
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs /media/darr/results/figures/brightness_temperature_error_timeseries
 
