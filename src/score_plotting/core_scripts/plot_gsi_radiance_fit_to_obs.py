#!/usr/bin/env python

"""
"""

import os
import pathlib
import warnings
import argparse
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from mpi4py import MPI

from score_plotting.core_scripts import gsistats_timeseries
from score_plotting.core_scripts.instrument_channel_nums import get_instrument_channels
from score_plotting.core_scripts.instrument_channel_nums import get_instrument_longnames
from score_plotting.core_scripts import satellite_names

HOURS_PER_DAY = 24. # hours

def config():
    """Add experiment name entries to experiment_list and
    experiment_plot_dict. The order of experiment_list is used
    to determine the order of plotting.
    """
    args = parse_arguments()
    if args.dark_theme:
        mpl_style_sheet = 'dark_theme.mplstyle'
    else:
        #mpl_style_sheet = 'full_3x3pg.mplstyle'
        mpl_style_sheet = 'ams_full.mplstyle'
        
    config_dict = {
        'config_path':
            os.path.join(pathlib.Path(__file__).parent.parent.resolve(),
                         'style_lib'),
        'config_file': [mpl_style_sheet],
        'output_path': args.figure_output_path,
        'experiment_list': ['cfsr',
                            'GDAS',
                            'NASA_GEOSIT_GSISTATS',
                            'replay_observer_diagnostic_v1.1',
                            'scout_run_v1',
                            '3dvar_coupledreanl_scoutrun_v1_test1',
                            '3dvar_coupledreanl_scoutrun_v2'
                         ],
        
        'experiment_plot_dict': {
            'cfsr' :
               {'color' : 'green',
                'ls': '-',
                'lw': 1.5
            },
            
            'NASA_GEOSIT_GSISTATS' :
                {'color' : '#E4002B',
                 'ls': '-',
                 'lw': 1.
            },
            'GDAS' : {
                'color' : '#0085CA',
                'ls': '-',
                'lw': 1.25
            },
            'replay_observer_diagnostic_v1.1' : {
                'color' : '#096FAE', #'#0A3758',
                'ls': '-',
                'lw': 0.75
            },
            'scout_run_v1' : {
                'color' : '#000000',
                'ls': '-',
                'lw': 0.5
            },
            'replay_observer_diagnostic_overlap' : {
                'color' : '#8D7334',
                'ls': '-',
                'lw': 0.5
            },
            '3dvar_coupledreanl_scoutrun_v1_test1' : {
                'color' : '#565A5C',
                'ls': '-',
                'lw': 0.5
            },
            '3dvar_coupledreanl_scoutrun_v2' : {
                'color' : '#8D7334',
                'ls': '-',
                'lw': 0.75
            }
        },
        'sensor_list': get_instrument_channels().keys(),
        'start_date': '2022-10-01 00:00:00',
        'stop_date': '2024-03-30 23:59:59',
    }
    
    '''
    could this be done by string matching for the std/bias etc part? we could
    have a basic friendly dict for that
    '''
    friendly_names_dict={'cfsr': "CFSR",
                         "scout_run_v1": "scout-atm (v0.01)",
                         "NASA_GEOSIT_GSISTATS": "GEOS-IT",
                         "GDAS": "GDAS",
                         "replay_observer_diagnostic_v1.1": "Replay",
                         "replay_observer_diagnostic_overlap": "UFS-replay-overlap",
                         "3dvar_coupledreanl_scoutrun_v1_test1": "scout-1 (v0.1)",
                         '3dvar_coupledreanl_scoutrun_v2': "scout-2 (v0.21)",
                         "std_GSIstage_1": "STD",
                         "variance_GSIstage_1": "obs error variance",
                         "bias_post_corr_GSIstage_1": "ME",
                         "sqrt_bias_GSIstage_1": "RMSE"}
                         
    return(config_dict, friendly_names_dict)
    
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Script to create GSI analysis timeseries figures for '
                    'radiance error monitoring')
    
    # Make figure_output_path optional (defaults to $HOME)
    parser.add_argument('figure_output_path', type=str, nargs='?',
                        default=pathlib.Path.home(),
                        help='Path to where figures will be saved')
    
    # Add an argument for interactive plotting
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive plotting. If specified, plots will be displayed interactively.')
                        
    # Add optional argument for satellite
    parser.add_argument('--satellite', type=str, default='all',
                        help='Satellite name')
    
    # Add optional argument for channel
    parser.add_argument('--channel', type=int, default=9999,
                            help='Channel number (e.g., 1, 2, 3, ...)')
    
    # Add optional argument for sensor
    parser.add_argument('--sensor', type=str, default='all',
                        help='Sensor name (e.g., atms, amsua)')
                        
    parser.add_argument('--gsi_stage', type=int, default=1,
                        help='GSI analysis iteration')
                        
    # Add DA cycle as an argument (optional, default to 6.0)
    parser.add_argument('--da_cycle', type=float, default=6.,
                        help='The DA cycle duration in hours (default: 6.0)')
    
    # Add days to smooth as an argument (optional, default to 8.0)
    parser.add_argument('--days_to_smooth', type=float, default=8.,
                        help='Number of days to smooth (default: 8.0)')
                        
    parser.add_argument(
        '--dark_theme', 
        action='store_true',  # If this argument is provided, dark_theme will be True
        help="Enable dark theme (default is False)"
    )
    
    parser.add_argument(
        '--yaxis_obs_err_scaler',
        type=float,
        default=3.,
        help='Scale vertical limits of bias and RMSE plots by this amount (times the maximum obs error)',
        
    )
    parser.add_argument(
        '--rmse_legend',
        action='store_true',
        help='Enable RMSE legend'
    )
    
    args = parser.parse_args()

    return args

def get_data_frame(experiment_list, sensor_list,
                   start_date='1979-01-01 00:00:00',
                   stop_date='2026-01-01 00:00:00',
                   select_sat_name=False,
                   sat_name=None,
                   gsi_it=1):
        
    gsi_it = int(gsi_it)
    array_metric_list = list()
    for sensor in sensor_list:
        array_metric_list.append(f'{sensor}_bias_post_corr_GSIstage_{gsi_it}')
        array_metric_list.append(f'{sensor}_std_GSIstage_{gsi_it}')
        array_metric_list.append(f'{sensor}_variance_GSIstage_{gsi_it}')
        array_metric_list.append(f'{sensor}_sqrt_bias_GSIstage_{gsi_it}')
        array_metric_list.append(f'{sensor}_nobs_used_GSIstage_{gsi_it}')
        array_metric_list.append(f'{sensor}_nobs_tossed_GSIstage_{gsi_it}')
        array_metric_list.append(f'{sensor}_use_GSIstage_None')
        
    return gsistats_timeseries.get_data_frame(
                               experiment_list,
                               array_metric_list,
                               start_date=start_date,
                               stop_date=stop_date,
                               select_sat_name=select_sat_name,
                               sat_name=sat_name)

class GSIRadianceFit2ObsFig(object):
    """
    """
    def __init__(self, data_frame=None, input_data_frame=False,
                 gsi_it=1):
        """
        """
        self.gsi_it = int(gsi_it)
        self.config_dict, self.friendly_names_dict = config()
        self.channel_dict = get_instrument_channels()
        self.channels_to_plot = self.channel_dict
        self.sensor_longnames = get_instrument_longnames()
        self.experiment_list = self.config_dict['experiment_list']
                
        if input_data_frame:
            self.data_frame = data_frame
        else:
            self.data_frame = get_data_frame(self.experiment_list,
                                             self.config_dict['start_date'],
                                             self.config_dict['stop_date'],
                                             gsi_it=self.gsi_it)
    
    def config_figure_params(self, days_to_smooth=1.):
        if self.config_dict['config_path'] and self.config_dict['config_file']:
            for style_file in self.config_dict['config_file']:
                style_file_path = os.path.join(self.config_dict['config_path'],
                                               style_file)
                plt.style.use(style_file_path)
        
        if self.dark_theme:
            self.default_plot_color = '#CFB87C'
            self.fill_color = '#565A5C'
            if 'GDAS' in self.config_dict['experiment_plot_dict'].keys():
                self.config_dict['experiment_plot_dict']['GDAS']['color'] = 'white'

            for expt_name in self.config_dict['experiment_plot_dict'].keys():
                if self.config_dict['experiment_plot_dict'][expt_name]['color'] == 'black':
                    self.config_dict['experiment_plot_dict'][expt_name]['color'] = self.default_plot_color
                elif self.config_dict['experiment_plot_dict'][expt_name]['color'] == '#003087':
                    self.config_dict['experiment_plot_dict'][expt_name]['color'] = 'white'
        else:
            self.default_plot_color = 'black'
            self.fill_color = '#A2A4A3'
            for expt_name in self.config_dict['experiment_plot_dict'].keys():
                if self.config_dict['experiment_plot_dict'][expt_name]['color'] == '#CFB87C':
                    self.config_dict['experiment_plot_dict'][expt_name]['color'] = self.default_plot_color
        
        self.window_size = pd.Timedelta(hours=HOURS_PER_DAY * days_to_smooth)
        self.min_periods = int(
            np.around(days_to_smooth * (HOURS_PER_DAY / self.da_cycle))
        )        
        
        self.time_domain = pd.Series(
            data = np.nan,
            index = pd.date_range(
                start = datetime.strptime(
                    self.config_dict['start_date'], '%Y-%m-%d %H:%M:%S'
                ),
                end = datetime.strptime(
                    self.config_dict['stop_date'], '%Y-%m-%d %H:%M:%S'
                ),
                freq = pd.Timedelta(hours = self.da_cycle)
            )
        )
    
    def build_timeseries(self, interactive_figure=False,
                         da_cycle = 6., # hours
                         days_to_smooth = 1., # days
                         dark_theme=False,
                         yaxis_obs_err_scaler=3.,
                         rmse_legend=False):
        self.dark_theme = dark_theme
        self.da_cycle = da_cycle
        self.config_figure_params(days_to_smooth=days_to_smooth)
        self.yaxis_obs_err_scaler = yaxis_obs_err_scaler
        self.rmse_legend=rmse_legend
        
        for sensor, channel_list in self.channel_dict.items():
            if sensor in self.config_dict['sensor_list']:
                experiment_timeseries_datetime_init=None
                array_metric_list = [f'{sensor}_bias_post_corr_GSIstage_{self.gsi_it}',
                                     f'{sensor}_std_GSIstage_{self.gsi_it}',
                                     f'{sensor}_variance_GSIstage_{self.gsi_it}',
                                     f'{sensor}_sqrt_bias_GSIstage_{self.gsi_it}',
                                     f'{sensor}_nobs_used_GSIstage_{self.gsi_it}',
                                     f'{sensor}_nobs_tossed_GSIstage_{self.gsi_it}',
                                     f'{sensor}_use_GSIstage_None'
                                 ]
                                     
                self.experiment_timeseries_dict = dict()
                for experiment in self.experiment_list:
                    self.experiment_timeseries_dict[experiment] = dict()
                    for array_metric in array_metric_list:
                        try:
                            self.experiment_timeseries_dict[
                                experiment][array_metric] = gsistats_timeseries.GSIStatsTimeSeries(
                                                self.config_dict['start_date'],
                                                self.config_dict['stop_date'],
                                                data_frame=self.data_frame,
                                                input_data_frame=True,
                                                experiment_name=experiment,
                                                array_metric_types=array_metric)
                            experiment_timeseries_datetime_init = self.experiment_timeseries_dict[
                                experiment][array_metric].init_datetime
                            self.experiment_timeseries_dict[experiment][array_metric].build()
                        except KeyError: # remove array_metric from dict if no records returned
                            self.experiment_timeseries_dict[experiment].pop(
                                array_metric, None)
                            warnings.warn(f'missing {sensor} records for '
                                          f'{experiment} experiment: {array_metric}')
                
                self.db_name = os.getenv('SCORE_POSTGRESQL_DB_NAME')        
                self.make_figures(
                    sensor,
                    init_datetime=experiment_timeseries_datetime_init,
                    interactive=interactive_figure)
    
    def make_figures(self, sensor, ncols=3, init_datetime=None,
                     alpha_foreground=0.9,
                     alpha_background=0.1,
                     interactive=False):
        
        sensor_longname = self.sensor_longnames[sensor]
        
        output_dir = os.path.join(self.config_dict['output_path'], f"{sensor}")
        locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
        formatter = mdates.ConciseDateFormatter(locator)
        
        month_locator = mdates.MonthLocator(interval=1)
        
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        nrows = 0
        sat_set = set()
        for experiment, timeseries_dict in self.experiment_timeseries_dict.items():
            for full_stat_name, timeseries_data in timeseries_dict.items():
                for stat_label, value_dict in timeseries_data.value_dict.items():
                    for sat_sensor in value_dict.keys():
                       sat_set.add(sat_sensor)
        
        for i, channel_num in enumerate(self.channels_to_plot[sensor]):
            channel_idx = np.argmin(np.abs(np.array(self.channel_dict[sensor]) - channel_num))
            if len(sat_set) > 0:
                max_yerr=0.5 # temperature (K)
                figsize_width = 4.5 * ncols
                figsize_length = 4.53 * len(sat_set)
                fig, axes = plt.subplots(len(sat_set), ncols, sharex=True,sharey=False,
                                         squeeze=False,
                                         figsize=(figsize_width, figsize_length))
                
                if self.gsi_it == 1:
                    difference_str = "Ob - Bg"
                elif self.gsi_it >= 2:
                    difference_str = "Ob - Anal"
                
                title_str0 = f"GSI radiance data anal fit to obs ({difference_str}) [metrics downloaded: {self.db_name}"
                
                if init_datetime:
                    init_ctime = init_datetime.ctime()
                    title_str1 = f" {init_ctime}]"
                
                else:
                    title_str1 = "]"
                    
                #fig.suptitle(f"{title_str0}{title_str1}")
                #axes[-1, 0].set_xlabel = 'Cycle date (Gregorian)'
                #axes[-1, 1].set_xlabel = 'Cycle date (Gregorian)'
                #axes[-1, 2].set_xlabel = 'Cycle date (Gregorian)'
            
                for row, sat_sensor in enumerate(sorted(sat_set)):
                    sat_short_name = sat_sensor.split('_')[:-1][0]
                    sat_label = satellite_names.get_longname(sat_short_name)
                    
                    # subplot titles
                    axes[row, 0].set_title(f"Mean {difference_str}: {sensor_longname} chan {channel_num} ({sat_label})")
                    axes[row, 1].set_title(f"RMS {difference_str}: {sensor_longname} chan {channel_num} ({sat_label})")
                    axes[row, 2].set_title(f"Nobs used: {sensor_longname} chan {channel_num} ({sat_label})")
                    
                    # vertical axes labels
                    axes[row, 0].set_ylabel(f'Brightness temp mean {difference_str} '
                                            r'($^{\circ}$C)')
                    axes[row, 1].set_ylabel(f'Brightness temp RMS {difference_str} '
                                            '($^{\circ}$C)')
                    axes[row, 2].set_ylabel(f'Number of obs used (+) and tossed (-)')
                    #rejection_ratio_ax = axes[row, 2].twinx()
                    #rejection_ratio_ax.set_ylabel('Percentage of observations tossed (%)')
                    if self.dark_theme:
                        axes[row, 0].axhline(color='#A2A4A3', lw=0.75)
                        axes[row, 2].axhline(color='#A2A4A3', lw=0.75)
                    else:
                        axes[row, 0].axhline(color='black', lw=0.75)
                        axes[row, 2].axhline(color='black', lw=0.75)

                    #rejection_ratio_ax.set_ylim(0, 100)
                    #rejection_ratio_ax.set_yticks(np.arange(0, 100.1, 20))                    
                    #rejection_ratio_ax.set_yticks(np.arange(0, 100.1, 5), minor=True)

                    # Set ticks on both left and right vertical axes
                    axes[row, 0].tick_params(axis='y', which='both', left=True, right=True)
                    axes[row, 1].tick_params(axis='y', which='both', left=True, right=True)
                    axes[row, 2].tick_params(axis='y', which='both', left=True, right=True)
                    
                    axes[row, 0].tick_params(axis='x', which='both', top=True, bottom=True,
                                             labelbottom=True)
                    axes[row, 1].tick_params(axis='x', which='both', top=True, bottom=True,
                                             labelbottom=True)
                    axes[row, 2].tick_params(axis='x', which='both', top=True, bottom=True,
                                             labelbottom=True)
                
                    #experiment_idx = 0
                    for experiment, timeseries_dict in self.experiment_timeseries_dict.items():
                        for full_stat_name, timeseries_data in timeseries_dict.items():
                            for stat_label, value_dict in timeseries_data.value_dict.items():
                                if stat_label == f'bias_post_corr_GSIstage_{self.gsi_it}' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    """ mean error plot
                                    """
                                    bias_timeseries = pd.Series(
                                        data=np.array(
                                            value_dict[sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    ).astype(float)
                                    
                                    std_timeseries = pd.Series(
                                        data=np.array(
                                            timeseries_dict[f'{sensor}_std_GSIstage_{self.gsi_it}'].value_dict[
                                                f'std_GSIstage_{self.gsi_it}'][sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_dict[f'{sensor}_std_GSIstage_{self.gsi_it}'].timestamp_dict[
                                            f'std_GSIstage_{self.gsi_it}'][sat_sensor]
                                    ).astype(float)
                                    
                                    nobs_used_timeseries = pd.Series(
                                        data=np.array(
                                            timeseries_dict[f'{sensor}_nobs_used_GSIstage_{self.gsi_it}'].value_dict[
                                                f'nobs_used_GSIstage_{self.gsi_it}'][sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_dict[f'{sensor}_nobs_used_GSIstage_{self.gsi_it}'].timestamp_dict[
                                            f'nobs_used_GSIstage_{self.gsi_it}'][sat_sensor]
                                    ).astype(float)
                                    
                                    use_flag_timeseries = pd.Series(
                                        data=np.array(
                                            timeseries_dict[
                                                f'{sensor}_use_GSIstage_None'].value_dict[
                                                    'use_GSIstage_None'][sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_dict[f'{sensor}_use_GSIstage_None'].timestamp_dict[
                                                'use_GSIstage_None'][sat_sensor]
                                    ).astype(float)

                                    standard_errs = std_timeseries / np.sqrt(nobs_used_timeseries)

                                    bias_timeseries = bias_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    use_flag_timeseries = use_flag_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    
                                    mean_values_smooth = bias_timeseries.rolling(
                                        window=self.window_size,
                                        min_periods=self.min_periods,
                                        center=True,
                                        #win_type='triang'
                                    ).mean()
                                    
                                    yerr_bot = (
                                        bias_timeseries - standard_errs
                                    ).combine_first(self.time_domain)
                                    yerr_top = (
                                        bias_timeseries + standard_errs
                                    ).combine_first(self.time_domain)
                                    
                                    axes[row, 0].fill_between(
                                        yerr_bot.index,
                                        yerr_top.values,
                                        y2=yerr_bot.values,
                                        lw=0,
                                        edgecolor='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_background,
                                        zorder=2
                                    )
                                
                                    '''
                                    axes[row, 0].plot(
                                        bias_timeseries.index,
                                        bias_timeseries.values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_background,
                                        lw=0.5,
                                        ls='-',
                                    )
                                    '''
                                    
                                    axes[row, 0].plot(
                                        mean_values_smooth.index,
                                        mean_values_smooth.where(
                                            use_flag_timeseries < 1
                                        ).values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_foreground,
                                        lw=self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=':',
                                       zorder=3, #label=self.friendly_names_dict[experiment]
                                    )
                                    
                                    axes[row, 0].plot(
                                        mean_values_smooth.index,
                                        mean_values_smooth.where(
                                            use_flag_timeseries > 0
                                        ).values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=1.0,
                                        lw=4.*self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=self.config_dict['experiment_plot_dict']
                                            [experiment]['ls'],
                                        label=self.friendly_names_dict[experiment],
                                        zorder=3
                                    )

                                    axes[row,0].legend(loc='upper right')
                            
                                elif stat_label == f'sqrt_bias_GSIstage_{self.gsi_it}' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    """RMS error plot
                                    """
                                    rmse_timeseries = pd.Series(
                                        data=np.array(
                                            value_dict[sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    ).astype(float)
                                    
                                    obs_err_var_timeseries = pd.Series(
                                        data=np.array(
                                            timeseries_dict[f'{sensor}_variance_GSIstage_{self.gsi_it}'].value_dict[
                                                f'variance_GSIstage_{self.gsi_it}'][sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_dict[f'{sensor}_variance_GSIstage_{self.gsi_it}'].timestamp_dict[
                                            f'variance_GSIstage_{self.gsi_it}'][sat_sensor]
                                    ).astype(float)
                                    
                                    use_flag_timeseries = pd.Series(
                                        data=np.array(
                                            timeseries_dict[
                                                f'{sensor}_use_GSIstage_None'].value_dict[
                                                    'use_GSIstage_None'][sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_dict[f'{sensor}_use_GSIstage_None'].timestamp_dict[
                                                'use_GSIstage_None'][sat_sensor]
                                    ).astype(float)

                                    max_yerr = np.max(
                                        np.nan_to_num(
                                            np.sqrt(obs_err_var_timeseries.values)
                                        ),
                                        initial=max_yerr
                                    )
                                    
                                    rmse_timeseries = rmse_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    use_flag_timeseries = use_flag_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    
                                    rmse_values_smooth = rmse_timeseries.rolling(
                                        window=self.window_size,
                                        min_periods=self.min_periods,
                                        center=True,
                                        #win_type='triang'
                                    ).mean()
                                
                                    axes[row, 1].plot(
                                        rmse_timeseries.index,
                                        rmse_timeseries.values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_background,
                                        lw=0.5,
                                        ls='-',
                                        zorder=2
                                    )
                                    
                                    axes[row, 1].plot(
                                        rmse_values_smooth.index,
                                        rmse_values_smooth.where(
                                            use_flag_timeseries < 1
                                        ).values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_foreground,
                                        lw=self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=':',
                                        zorder=3
                                        #label=self.friendly_names_dict[experiment],
                                    )
                                    
                                    axes[row, 1].plot(
                                        rmse_values_smooth.index,
                                        rmse_values_smooth.where(
                                            use_flag_timeseries > 0
                                        ),
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=1.0,
                                        lw=4.*self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=self.config_dict['experiment_plot_dict']
                                            [experiment]['ls'],
                                        label=self.friendly_names_dict[experiment],
                                        zorder=3
                                    )
                                    
                                    if self.rmse_legend:
                                        axes[row,1].legend(loc='lower left')
                            
                                elif stat_label == f'nobs_used_GSIstage_{self.gsi_it}' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    """ nobs tossed and rejection ratio plot
                                    """
                                    nobs_used_timeseries = pd.Series(
                                        data=np.array(
                                            value_dict[sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    ).astype(float)

                                    use_flag_timeseries = pd.Series(
                                        data=np.array(
                                            timeseries_dict[
                                                f'{sensor}_use_GSIstage_None'].value_dict[
                                                    'use_GSIstage_None'][sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_dict[f'{sensor}_use_GSIstage_None'].timestamp_dict[
                                                'use_GSIstage_None'][sat_sensor]
                                    ).astype(float)

                                    nobs_used_timeseries = nobs_used_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    
                                    use_flag_timeseries = use_flag_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    
                                    nobs_used_smooth = nobs_used_timeseries.rolling(
                                        window=self.window_size,
                                        min_periods=self.min_periods,
                                        center=True,
                                        #win_type='triang'
                                    ).mean()
                                    
                                    axes[row, 2].plot(
                                        nobs_used_timeseries.index,
                                        nobs_used_timeseries.values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_background,
                                        lw=0.5,
                                        ls='-',
                                        zorder=2
                                    )
                                    
                                    axes[row, 2].plot(
                                        nobs_used_smooth.index,
                                        nobs_used_smooth.where(
                                            use_flag_timeseries < 1
                                        ).values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_foreground,
                                        lw=self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=':',
                                        zorder=3
                                        #label=f"{self.friendly_names_dict[experiment]}"
                                    )
                                    
                                    axes[row, 2].plot(
                                        nobs_used_smooth.index,
                                        nobs_used_smooth.where(
                                            use_flag_timeseries > 0
                                        ).values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=1.0,
                                        lw=4.*self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=self.config_dict['experiment_plot_dict']
                                            [experiment]['ls'],
                                        label=f"{self.friendly_names_dict[experiment]}",
                                        zorder=3
                                    )
                                    
                                    #axes[row,2].legend(loc='lower right')
                                    #rejection_ratio_ax.legend(loc='upper right')
                                    
                                elif stat_label == f'nobs_tossed_GSIstage_{self.gsi_it}' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    """ nobs tossed and rejection ratio plot
                                    """
                                    nobs_tossed_timeseries = pd.Series(
                                        data=np.array(
                                            value_dict[sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    ).astype(float)

                                    use_flag_timeseries = pd.Series(
                                        data=np.array(
                                            timeseries_dict[
                                                f'{sensor}_use_GSIstage_None'].value_dict[
                                                    'use_GSIstage_None'][sat_sensor]
                                        )[:, channel_idx],
                                        index=timeseries_dict[f'{sensor}_use_GSIstage_None'].timestamp_dict[
                                                'use_GSIstage_None'][sat_sensor]
                                    ).astype(float)

                                    nobs_tossed_timeseries = nobs_tossed_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    
                                    use_flag_timeseries = use_flag_timeseries.combine_first(
                                        self.time_domain
                                    )
                                    
                                    nobs_tossed_smooth = nobs_tossed_timeseries.rolling(
                                        window=self.window_size,
                                        min_periods=self.min_periods,
                                        center=True,
                                        #win_type='triang'
                                    ).mean()
                                    
                                    axes[row, 2].plot(
                                        nobs_tossed_timeseries.index,
                                        -1. * nobs_tossed_timeseries.values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_background,
                                        lw=0.5,
                                        ls='-',
                                        zorder=2
                                    )
                                    
                                    axes[row, 2].plot(
                                        nobs_tossed_smooth.index,
                                        -1. * nobs_tossed_smooth.where(
                                            use_flag_timeseries < 1
                                        ).values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=alpha_foreground,
                                        lw=self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=':',
                                        zorder=3
                                        #label=f"{self.friendly_names_dict[experiment]}"
                                    )
                                    
                                    axes[row, 2].plot(
                                        nobs_tossed_smooth.index,
                                        -1. * nobs_tossed_smooth.where(
                                            use_flag_timeseries > 0
                                        ).values,
                                        marker='none',
                                        color=self.config_dict['experiment_plot_dict']
                                            [experiment]['color'],
                                        alpha=1.0,
                                        lw=4.*self.config_dict['experiment_plot_dict']
                                            [experiment]['lw'],
                                        ls=self.config_dict['experiment_plot_dict']
                                            [experiment]['ls'],
                                        label=f"{self.friendly_names_dict[experiment]}",
                                        zorder=3
                                    )

                                for col_idx in range(ncols):
                                    if self.dark_theme:
                                        axes[row, col_idx].grid(True)
                                    axes[row, col_idx].xaxis.set_major_locator(locator)
                                    axes[row, col_idx].xaxis.set_major_formatter(formatter)
                                    axes[row, col_idx].xaxis.set_minor_locator(month_locator)
                                          
                        #experiment_idx += 1
                
                for row, sat_sensor in enumerate(sorted(sat_set)):
                    # set ylim, ticks                        
                    if self.yaxis_obs_err_scaler < 3.:
                        axes[row, 0].set_yticks(
                            np.arange(np.around((-self.yaxis_obs_err_scaler/6.) * max_yerr - 0.2, decimals=1),
                                      (self.yaxis_obs_err_scaler/6.) * max_yerr + 0.2,
                                      0.01),
                        )
                    
                    elif max_yerr < 1:
                        axes[row, 0].set_yticks(
                            np.arange(np.around((-self.yaxis_obs_err_scaler/6.) * max_yerr - 0.2, decimals=1),
                                      (self.yaxis_obs_err_scaler/6.) * max_yerr + 0.2,
                                      0.1),
                        )
                    else:
                        axes[row, 0].set_yticks(
                            np.arange(np.around((-self.yaxis_obs_err_scaler/6.) * max_yerr - 1),
                                      (self.yaxis_obs_err_scaler/6.) * max_yerr + 1,
                                      0.5),
                        )
                        axes[row, 0].set_yticks(
                            np.arange(np.around((-self.yaxis_obs_err_scaler/6.) * max_yerr - 0.2, decimals=1),
                                      (self.yaxis_obs_err_scaler/6.) * max_yerr + 0.2,
                                      0.1),
                                      minor=True
                    )
                    
                    if self.yaxis_obs_err_scaler < 3.:
                        axes[row, 1].set_yticks(
                            np.arange(0, self.yaxis_obs_err_scaler * max_yerr + 0.1, 0.01),
                            minor=True
                        )
                    
                        axes[row, 1].set_yticks(
                            np.arange(0, self.yaxis_obs_err_scaler * max_yerr + 1, 0.1)
                        )
                    else:
                        axes[row, 1].set_yticks(
                            np.arange(0, self.yaxis_obs_err_scaler * max_yerr + 0.1, 0.1),
                            minor=True
                        )
                    
                        axes[row, 1].set_yticks(
                            np.arange(0, self.yaxis_obs_err_scaler * max_yerr + 1, 0.5)
                        )
                    
                    axes[row, 0].set_ylim((-self.yaxis_obs_err_scaler/6.)*max_yerr, (self.yaxis_obs_err_scaler/6.)*max_yerr)
                    axes[row, 1].set_ylim(0, self.yaxis_obs_err_scaler*max_yerr)
                    
                    nobs_ylims = axes[row, 2].get_ylim()
                    
                    """"
                    if nobs_ylims[0] < 0:
                        axes[row, 2].set_ylim(bottom=0)
                    """
                
                fig.suptitle(f"{title_str0}{title_str1}")
                plt.tight_layout()
                plt.subplots_adjust(top = 1. - 1.2 / figsize_length)
                if interactive:
                    plt.show()
                else:
                    if self.gsi_it ==1:
                        fig_title=f'gsi_radiance_{sensor}_ch{channel_num}_omb.png'
                    elif self.gsi_it >=2:
                        fig_title=f'gsi_radiance_{sensor}_ch{channel_num}_oma.png'
                    plt.savefig(os.path.join(output_dir, fig_title), dpi=300)
                plt.close()

def run_microwave_sounders(sensor_list=['amsua', 'amsub', 'atms', 'ssmi', 'ssmis']):
    prun(sensor_list=sensor_list)
    
def run_microwave_sounders2(sensor_list=['amsua', 'amsub', 'atms', 'ssmi', 
                                         'ssmis','hirs2', 'hirs3', 'hirs4',
                                         'ssu', 'msu']):
    prun(sensor_list=sensor_list)

def run_atms(sensor_list=['atms']):
    prun(sensor_list=sensor_list)

def run_airs(sensor_list=['airs']):
    prun(sensor_list=sensor_list)

def run_tovs(sensor_list = ['hirs2', 'hirs3', 'hirs4', 'ssu', 'msu']):
    prun(sensor_list = sensor_list)

def run_avhrr(sensor_list = ['avhrr2', 'avhrr3']):
    prun(sensor_list=sensor_list)

def prun(experiment_list=None, sensor_list=None, start_date=None, stop_date=None):
    args = parse_arguments()
    gsi_it = args.gsi_stage
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load global configurations and friendly names
    global_config_dict, global_friendly_names_dict = config()
    
    if experiment_list is None:
        experiment_list = global_config_dict['experiment_list']
        
    if start_date is None:
        start_date = global_config_dict['start_date']
        
    if stop_date is None:
        stop_date = global_config_dict['stop_date']
    
    if args.sensor != 'all':
        sensor_list = [args.sensor]
    
    if sensor_list is None:
        sensor_list = list()
        for sensor in global_config_dict['sensor_list']:
            sensor_list.append(sensor)
            
    if args.satellite != 'all':
        select_sat_name = True
    else:
        select_sat_name = False

    # Rank 0 prepares the data
    if rank == 0:
        global_data_frame = get_data_frame(
            experiment_list,
            sensor_list,
            start_date=start_date,
            stop_date=stop_date,
            select_sat_name = select_sat_name,
            sat_name=args.satellite,
            gsi_it=gsi_it)

        # Split the data by sensor (one part per sensor)
        data_frame_parts_dict = dict()
        for sensor in sensor_list:
            data_frame_parts_dict[sensor] = global_data_frame[
                global_data_frame['metric_instrument_name'] == sensor]

    else:
        data_frame_parts_dict = None

    # Calculate how many sensors each process should handle
    sensors_per_process = len(sensor_list) // size

    # Handle leftover sensors (remaining sensors are distributed to the first few processes)
    leftover_sensors = len(sensor_list) % size
    
    if rank == 0:
        for i in range(0, size):
            # Calculate the subset of sensors for this rank
            start_idx = i * sensors_per_process + min(i, leftover_sensors)
            end_idx = start_idx + sensors_per_process + (1 if i < leftover_sensors else 0)
            rank_sensors = sensor_list[start_idx:end_idx]
            
            # Prepare the data for this rank
            data_to_send = {sensor: data_frame_parts_dict[sensor] for sensor in rank_sensors}

            if i==0:
                local_data_frames = data_to_send
            else:
                comm.send(data_to_send, dest=i, tag=11+i)
    else:
        local_data_frames = comm.recv(source=0, tag=11+rank)

    # Each process works on its part of the data
    if local_data_frames is not None:
        for sensor, data_frame in local_data_frames.items():
            # Only work on data for the specific sensor assigned to the process
            '''
            print(f"Rank {rank} is processing sensor: {sensor} and here is "
                   f"the data frame: {data_frame.metric_instrument_name}")
            
            '''
            experiment_metrics_timeseries_data = GSIRadianceFit2ObsFig(
                data_frame=data_frame,
                input_data_frame=True,
                gsi_it=gsi_it
            )
            if args.channel != 9999:
                experiment_metrics_timeseries_data.channels_to_plot = {sensor: [args.channel]}
            experiment_metrics_timeseries_data.config_dict['sensor_list'] = [sensor]
            experiment_metrics_timeseries_data.experiment_list = experiment_list
            experiment_metrics_timeseries_data.config_dict['start_date'] = start_date
            experiment_metrics_timeseries_data.config_dict['stop_date'] = stop_date
            experiment_metrics_timeseries_data.build_timeseries(interactive_figure=args.interactive,
                                                                days_to_smooth=args.days_to_smooth,
                                                                da_cycle=args.da_cycle,
                                                                dark_theme=args.dark_theme,
                                                                yaxis_obs_err_scaler = args.yaxis_obs_err_scaler,
                                                                rmse_legend=args.rmse_legend)

def main():
    """
    """
    #run_avhrr()
    #run_tovs()
    #run_microwave_sounders2()
    #run_atms()
    prun()

if __name__ == "__main__":
    main()
