#!/bin/bash
#SBATCH --job-name=gsi_radiance_plots      # Name of the job
#SBATCH --output=plot_gsi%j_out.log     # Output log file with job ID
#SBATCH --error=plot_gsi%j_err.log     
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=24:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=bigmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Adam.Schneider@noaa.gov  # Email address for notifications

source $HOME/.bash_profile
micromamba activate darr_score_sqlalchemy1_dev_env
#which mpirun
# Run the MPI Python script using mpiexec

# commented out sensors stopped working January 2026

# 2023 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor viirs-m /media/darr/results/figures/brightness_temperature_error_timeseries
#mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor abi /media/darr/results/figures/brightness_temperature_error_timeseries
#mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor ahi /media/darr/results/figures/brightness_temperature_error_timeseries
#mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor seviri /media/darr/results/figures/brightness_temperature_error_timeseries
#mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor imgr /media/darr/results/figures/brightness_temperature_error_timeseries
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor mws /media/darr/results/figures/brightness_temperature_error_timeseries
#mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor cris /media/darr/results/figures/brightness_temperature_error_timeseries
#mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor cris-fsr /media/darr/results/figures/brightness_temperature_error_timeseries
#mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs --sensor iasi /media/darr/results/figures/brightness_temperature_error_timeseries

# 1997 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor viirs-m /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor mws /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout

# 1993 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2_1993 --sensor viirs-m /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1993
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2_1993 --sensor mws /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1993

# 1988 stream
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2_1988 --sensor viirs-m /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1988
mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2_1988 --sensor mws /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout_1988