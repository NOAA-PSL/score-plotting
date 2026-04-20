#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for a subset of sensors for the replay overlap experiment
"""

from score_plotting.core_scripts import plot_gsi_conv_fit_to_obs

def run(experiment_list=None, variable_list=None,
        start_date='1978-10-01 00:00:00',
        stop_date='2028-09-30 23:59:59'):

    plot_gsi_conv_fit_to_obs.prun(
        # Default is for fit of surface pressure data (hPa)
        start_date=start_date,
        stop_date=stop_date,
    )
    
    plot_gsi_conv_fit_to_obs.prun(
        variable_list=variable_list,
        experiment_list=experiment_list,
        start_date=start_date,
        stop_date=stop_date,
        pressure_bins=True
    )

def main():
    experiment_list=[
        '3dvar_coupledreanl_scoutrun_v2',
        'replay_observer_diagnostic_v1.1',
        #'NASA_GEOSIT_GSISTATS',
        #'cfsr',
        #'GDAS',
        #'replay_observer_diagnostic_v1',
        #'replay_observer_diagnostic_overlap',
        #'scout_run_v1',
        #'3dvar_coupledreanl_scoutrun_1979streamv1_test1',
        ]
    
    variable_list=[
        #'fit_psfc_data', # fit of surface pressure data (hPa)
        'fit_uv_data', # fit of u, v wind data (m/s)
        'fit_t_data', # fit of temperature data (K)
        'fit_q_data', # fit of moisture data (% of qsaturation guess)
    ]
    
    run(experiment_list=experiment_list, variable_list=variable_list,
        start_date='1996-10-01 00:00:00', stop_date='1997-09-30 23:59:59')
    run(experiment_list=experiment_list, variable_list=variable_list,
        start_date='2022-10-01 00:00:00', stop_date='2023-09-30 23:59:59')

if __name__=='__main__':
    main()