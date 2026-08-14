#!/bin/bash
#SBATCH --job-name=gdas_ocean_diags_plots      # Name of the job
#SBATCH --output=plot_gdas_ocean_diags%j.log     # Output log file with job ID
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=3:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=medmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Chesley.Mccoll@noaa.gov  # Email address for notifications

source $HOME/.bash_profile
micromamba activate darr_score_sqlalchemy1_dev_env
#which mpirun
# Run the MPI Python script using mpiexec
mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs /media/darr/results/figures/gdas_ocean_diags_error_timeseries

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_2023 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_2023

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_2018 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_2018

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_2013 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_2013

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_2008 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_2008

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_2003 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_2003

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_1997 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_1997

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_1993 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_1993

mpiexec python -m score_plotting.core_scripts.plot_soca_diags_fit_to_obs_1988 /media/darr/results/figures/gdas_ocean_diags_error_timeseries_1988