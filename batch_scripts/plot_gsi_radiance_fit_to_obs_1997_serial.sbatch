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
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor msu /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor gmi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor mhs /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor saphir /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor sndr /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor sndrD1 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor sndrD2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor sndrD3 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor sndrD4 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor avhrr2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor amsre /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor amsr2 /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor tmi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor abi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor ahi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor iasi /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor seviri /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor imgr /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor cris /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
timeout 3600 mpiexec python -m score_plotting.core_scripts.plot_gsi_radiance_fit_to_obs_rod --sensor cris-fsr /media/darr/results/figures/brightness_temperature_error_timeseries_weakly_coupled_scout
