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
            'replay_observer_diagnostic_v1',
            'replay_observer_diagnostic_overlap'
        ],
        start_date='2018-10-01 00:00:00',
        stop_date='2024-09-30 00:00:00'
    )

if __name__=='__main__':
    main()