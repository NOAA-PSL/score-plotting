#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for a subset of sensors for the replay overlap experiment
"""

from score_plotting.core_scripts import plot_gsi_conv_fit_to_obs_scout_v2

def main():
    plot_gsi_conv_fit_to_obs_scout_v2.run(start_date='2012-10-01 00:00:00', stop_date='2014-03-30 23:59:59')

if __name__=='__main__':
    main()