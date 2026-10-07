#!/bin/bash
#SBATCH --job-name=scout_movies      # Name of the job
#SBATCH --output=scout_movies%j.log     # Output log file with job ID
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2           # Number of tasks (CPUs) per node
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1                      # Number of nodes (adjust based on your requirements)
#SBATCH --time=1:00:00                # Max wall time (adjust as needed)
#SBATCH --partition=medmem #batch              # Partition to use
#SBATCH --mail-type=ALL                # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=Adam.Schneider@noaa.gov  # Email address for notifications

source $HOME/.bash_profile
micromamba activate darr_score_sqlalchemy1_dev_env

mv /media/darr/results/*.mpg /media/darr/results/movies/

python /media/darr/src/score-monitoring/scripts/make_movie.py -o=/media/darr/results/fv3pwv.mpg /media/darr/results/cylc-run/score-monitoring/cylc_scout_v2/scout_*_install2/run*/work/*/fv3gfs_sfc_mapper/fv3pwv_*.png
python /media/darr/src/score-monitoring/scripts/make_movie.py -o=/media/darr/results/fv3nhsic.mpg /media/darr/results/cylc-run/score-monitoring/cylc_scout_v2/scout_*_install2/run*/work/*/fv3gfs_sfc_mapper/fv3nhsic_*.png
python /media/darr/src/score-monitoring/scripts/make_movie.py -o=/media/darr/results/fv3shsic.mpg /media/darr/results/cylc-run/score-monitoring/cylc_scout_v2/scout_*_install2/run*/work/*/fv3gfs_sfc_mapper/fv3shsic_*.png
python /media/darr/src/score-monitoring/scripts/make_movie.py -o=/media/darr/results/fv3nhsitk.mpg /media/darr/results/cylc-run/score-monitoring/cylc_scout_v2/scout_*_install2/run*/work/*/fv3gfs_sfc_mapper/fv3nhsitk_*.png
python /media/darr/src/score-monitoring/scripts/make_movie.py -o=/media/darr/results/fv3shsitk.mpg /media/darr/results/cylc-run/score-monitoring/cylc_scout_v2/scout_*_install2/run*/work/*/fv3gfs_sfc_mapper/fv3shsitk_*.png
python /media/darr/src/score-monitoring/scripts/make_movie.py -o=/media/darr/results/soca_sst.mpg /media/darr/results/cylc-run/score-monitoring/cylc_scout_v2/scout_*_install2/run*/work/*/soca_diags_mapper/sst_*_sfc.png
