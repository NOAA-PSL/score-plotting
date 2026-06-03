#!/bin/bash
#SBATCH --job-name=gsi_conv_plots      # Name of the job
#SBATCH --output=plot_gsi_conv%j.log     # Output log file with job ID
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=12:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=bigmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Adam.Schneider@noaa.gov  # Email address for notifications

source $HOME/.bash_profile
micromamba activate darr_score_sqlalchemy1_dev_env

# Run the MPI Python script using mpiexec
mpiexec python -m score_plotting.core_scripts.plot_gsi_conv_fit_to_obs_scout_v2 /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries

cp /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries/type_120_gdas_gsi_conv_asm_fit_t_data_omb.png /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries/gdas_gsi_conv_asm_fit_t_data_omb.png
cp /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries/type_220_gdas_gsi_conv_asm_fit_uv_data_omb.png /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries/gdas_gsi_conv_asm_fit_uv_data_omb.png
cp /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries/type_120_gdas_gsi_conv_asm_fit_q_data_omb.png /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries/gdas_gsi_conv_asm_fit_q_data_omb.png

mpiexec python -m score_plotting.core_scripts.plot_gsi_conv_fit_to_obs_scout_v2_1988 /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries_1988
mpiexec python -m score_plotting.core_scripts.plot_gsi_conv_fit_to_obs_scout_v2_1993 /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries_1993
mpiexec python -m score_plotting.core_scripts.plot_gsi_conv_fit_to_obs_scout_v2_1997 /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries_1997
mpiexec python -m score_plotting.core_scripts.plot_gsi_conv_fit_to_obs_scout_v2_2023 /media/darr/results/figures/gdas_gsi_conv_fit2obs_error_timeseries_2023