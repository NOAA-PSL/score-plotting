#!/bin/bash
#SBATCH --job-name=gsi_radiance_plots_daily_update      # Name of the job
#SBATCH --output=plot_gsi%j.log     # Output log file with job ID
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=6:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=medmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Chesley.Mccoll@noaa.gov  # Email address for notifications

source $HOME/.bashrc
micromamba activate darr_score_sqlalchemy1_env
which mpirun
# Run the MPI Python script using srun
/media/darr/opt/micromamba/envs/darr_score_sqlalchemy1_env/bin/mpirun /media/darr/opt/micromamba/envs/darr_score_sqlalchemy1_env/bin/python /media/darr/src/score-plotting/src/score_plotting/core_scripts/plot_gsi_radiance_fit_to_obs_rod_overlap_daily_update.py /media/darr/results/figures/brightness_temperature_error_timeseries/overlap