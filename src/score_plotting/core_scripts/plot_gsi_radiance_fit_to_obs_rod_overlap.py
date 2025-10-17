#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for a subset of sensors for the replay overlap experiment
"""

from score_plotting.core_scripts import plot_gsi_radiance_fit_to_obs

def main():
    plot_gsi_radiance_fit_to_obs.prun(
        experiment_list=[
            'cfsr',
            'NASA_GEOSIT_GSISTATS',
            'GDAS',
            'replay_observer_diagnostic_v1',
            'replay_observer_diagnostic_overlap'
        ],
        start_date='2017-10-01 00:00:00',
        stop_date='2023-09-30 23:59:59'
    )

if __name__=='__main__':
    main()
