#!/usr/bin/env bash

python src/score_plotting/core_scripts/plot_gsi_radiance_fit_to_obs_rod_gdas_cfsr.py --days_to_smooth 30 --satellite 'NOAA 15' --channel 7 --rmse_legend --yaxis_obs_err_scaler 0.6
