#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for a subset of sensors for the replay overlap experiment
"""

from score_plotting.core_scripts import plot_gsi_radiance_fit_to_obs

def main():
    plot_gsi_radiance_fit_to_obs.prun(
        experiment_list=[
            'NASA_GEOSIT_GSISTATS',
            'GDAS',
            'replay_observer_diagnostic_v1.1',
            'cfsr',
            'scout_run_v1',
            #'3dvar_coupledreanl_scoutrun_1979streamv1_test1',
            '3dvar_coupledreanl_scoutrun_v1_test1'
        ],
        start_date='2022-10-01 00:00:00',
        stop_date='2023-10-01 00:00:00',
        sensor_list = [
            # microwave sounders:
            'amsua',
            'amsub',
            'atms',
            'ssmi',
            'ssmis',
            # infrared sounders:
            'airs',
            # TIROS operational vertical sounders (TOVS):
            'hirs2',
            'hirs3',
            'hirs4',
            'ssu',
            'msu',
            # Advanced Very-High-Resolution Radiometers:
            'avhrr2',
            'avhrr3'
        ]
        )

if __name__=='__main__':
    main()
