#!/usr/bin/env python

from score_plotting.core_scripts import plot_soca_diags_fit_to_obs 

def main():
    plot_soca_diags_fit_to_obs.main(start_date='1996-10-01 12:00:00',
                                    stop_date='2002-10-01 12:00:00')

if __name__=='__main__':
    main()