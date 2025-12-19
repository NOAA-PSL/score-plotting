#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for a subset of sensors for the replay overlap experiment
"""

from score_plotting.core_scripts import plot_gsi_radiance_fit_to_obs

def main():
    plot_gsi_radiance_fit_to_obs.prun(
        experiment_list=[
            'cfsr',
            'GDAS',
            'replay_observer_diagnostic_v1.1',
            #'scout_run_v1'
        ],
        sensor_list=['amsua']
    )

if __name__=='__main__':
    main()
