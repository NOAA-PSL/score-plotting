#!/bin/bash
#SBATCH --job-name=gsi_radiance_plots      # Name of the job
#SBATCH --output=plot_gsi%j.log     # Output log file with job ID
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=12:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=medmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Chesley.Mccoll@noaa.gov  # Email address for notifications

source $HOME/.bash_profile
micromamba activate darr_score_sqlalchemy1_dev_env
#which mpirun
# Run the MPI Python script using mpiexec
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor msu /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor gmi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor mhs /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor saphir /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor sndr /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor sndrD1 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor sndrD2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor sndrD3 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor sndrD4 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor avhrr2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor amsre /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor amsr2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor tmi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor abi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor ahi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor iasi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor seviri /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor imgr /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor cris /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_scout_v2 --sensor cris-fsr /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
