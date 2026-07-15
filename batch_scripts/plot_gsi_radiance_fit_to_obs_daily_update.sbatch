#!/bin/bash
#SBATCH --job-name=gsi_radiance_plots_daily_update      # Name of the job
#SBATCH --output=plot_gsi%j.log     # Output log file with job ID
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=24:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=medmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Adam.Schneider@noaa.gov  # Email address for notifications

source $HOME/.bash_profile
micromamba activate darr_score_sqlalchemy1_dev_env
#which mpirun
# Run the MPI Python script using mpiexec
# 2023 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update /media/darr/results/figures/brightness_temperature_error_timeseries
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries

# 2018 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_2018 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_2018
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_2018 --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_2018

# 2013 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_2013 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_2013
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_2013 --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_2013

# 1988 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_1988 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1988
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_1988 --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1988

# 1997 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2 --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout

mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v21 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1997_restart
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v21 --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1997_restart

# 1993 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_1993 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1993
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_1993 --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1993

# 1988 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_1988 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1988
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_daily_update_scout_v2_1988 --gsi_stage 2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1988
