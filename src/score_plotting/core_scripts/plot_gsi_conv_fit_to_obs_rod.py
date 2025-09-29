#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for a subset of sensors for the replay overlap experiment
"""

from score_plotting.core_scripts import plot_gsi_conv_fit_to_obs

def main():
    plot_gsi_conv_fit_to_obs.prun(
        experiment_list=[
            'NASA_GEOSIT_GSISTATS',
            'GDAS',
            'replay_observer_diagnostic_v1',
            'replay_observer_diagnostic_overlap',
            'scout_run_v1',
            '3dvar_coupledreanl_scoutrun_1979streamv1_test1',
            '3dvar_coupledreanl_scoutrun_v1_test1'
        ],
        start_date='1994-01-01 00:00:00',
        stop_date='2025-09-30 23:59:59'
    )

if __name__=='__main__':
    main()
