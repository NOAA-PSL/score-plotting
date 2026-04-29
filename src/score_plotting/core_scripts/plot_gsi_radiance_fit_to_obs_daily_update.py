#!/usr/bin/env python

""" wrapper to call plot_gsi_radiance_fit_to_obs figure
generation for subset of sensors to be updated daily
"""

from score_plotting.core_scripts import plot_gsi_radiance_fit_to_obs

def main():
    plot_gsi_radiance_fit_to_obs.prun(
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
            'avhrr3',
            # others
            'gmi',
            'mhs',
            'saphir',
            'sndr',
            'sndrD1',
            'sndrD2',
            'sndrD3',
            'sndrD4',
            'amsre',
            'amrs2',
            'tmi'
        ]
    )
    
if __name__=='__main__':
    main()
