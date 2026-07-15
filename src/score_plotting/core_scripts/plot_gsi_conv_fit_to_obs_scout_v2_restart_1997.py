#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for a subset of sensors for the replay overlap experiment
"""

from score_plotting.core_scripts import plot_gsi_conv_fit_to_obs_scout_v2

def main():
    plot_gsi_conv_fit_to_obs_scout_v2.run(experiment_list = [
            '3dvar_coupledreanl_scoutrun_restart_v21',
            '3dvar_coupledreanl_scoutrun_v2',
            #'replay_observer_diagnostic_v1.1',
            #'NASA_GEOSIT_GSISTATS',
            #'cfsr',
            #'GDAS',
            #'replay_observer_diagnostic_v1',
            #'replay_observer_diagnostic_overlap',
            #'scout_run_v1',
            #'3dvar_coupledreanl_scoutrun_1979streamv1_test1',
        ],
        start_date='1997-03-01 00:00:00', stop_date='1997-05-01 00:00:00')

if __name__=='__main__':
    main()
