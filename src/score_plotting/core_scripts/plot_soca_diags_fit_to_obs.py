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

from score_plotting.core_scripts import soca_diags_timeseries

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
        mpl_style_sheet = 'full_3x3pg.mplstyle'
        
    config_dict = {
        'config_path':
            os.path.join(pathlib.Path(__file__).parent.parent.resolve(),
                         'style_lib'),
        'config_file': [mpl_style_sheet],
        'output_path': args.figure_output_path,
        'experiment_list': [
            'replay_observer_diagnostic_v1.1',
            '3dvar_coupledreanl_scoutrun_v2',
                         ],
        
        'experiment_plot_dict': {
            '3dvar_coupledreanl_scoutrun_1979streamv1_test1' : {
                'color' : '#A2A4A3',
                'ls': '-',
                'lw': 1.5
            },
            '3dvar_coupledreanl_scoutrun_v1_test1' : {
                'color' : '#8D7334',
                'ls': '-',
                'lw': 0.75 
            },
        'replay_observer_diagnostic_v1.1' : {
            'color' : '#29cfed',#'#0A3758',
            'color2': '#29cfed',
            'marker': '+',
            'ls': '-',
            'lw': 0.5,
            'ls2': '-.',
            'zorder':2
            },
        '3dvar_coupledreanl_scoutrun_v2' : {
            'color' : '#9b8d62',
            'color2': '#9b8d62',
            'marker': '+',
            'ls': '-',
            'lw': 0.5,
            'ls2': '--',
            'zorder':2
            },
        },
        'region':'global',
        #'sensor_list': ['ctd', 'mbt', 'osd', 'xbt'],
        'sensor_list': ['avhrr', 'viirs', 'amsr2', 'ssmis'],
        'variable_list': ['seaSurfaceTemperature', 'seaIceFraction'],
        'units': {'seaSurfaceTemperature': r'$^{\circ}$C',
                  'seaIceFraction': 'unitless'},
        #'variable_list': ['waterTemperature'],
        'start_date': '1978-10-01 00:00:00',
        'stop_date': '2026-09-30 23:59:59',
    }
    
    '''
    could this be done by string matching for the std/bias etc part? we could
    have a basic friendly dict for that
    '''
    friendly_names_dict={"3dvar_coupledreanl_scoutrun_1979streamv1_test1": "weakly coupled 1979stream (3DVar)",
                         '3dvar_coupledreanl_scoutrun_v1_test1': "weakly coupled scout (3DVar)",
                         '3dvar_coupledreanl_scoutrun_v2': 'scout-2',
                         "osd": "ocean station data (OSD)",
                         "xbt": "expendable bathythermograph (XBT)",
                         "ctd": "conductivity-temperature-depth (CTD)",
                         "mbt": "mechanical bathythermograph (MBT)",
                         "avhrr": "Advanced Very-High Resolution Radiometer",
                         "viirs": "Visible Infrared Imaging Radiometer Suite",
                         "amsr2": "Advanced Microwave Scanning Radiometer 2",
                         "ssmis": "Special Sensor Microwave Imager/Sounder"}
                         
    return(config_dict, friendly_names_dict)
    
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Script to create SOCA diagnostics timeseries figures for '
                    'ocean monitoring')
    
    # Make figure_output_path optional (defaults to $HOME)
    parser.add_argument('figure_output_path', type=str, nargs='?',
                        default=pathlib.Path.home(),
                        help='Path to where figures will be saved')
    
    # Add an argument for interactive plotting
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive plotting. If specified, plots will be displayed interactively.')
    
    # Add optional argument for sensor
    parser.add_argument('--sensor', type=str, default='all',
                        help='Sensor name (e.g., ctd, xbt)')
                        
    # Add optional argument for variable
    parser.add_argument('--variable', type=str, default='all',
                        help='variable name (e.g., waterTemperature)')
                        
    parser.add_argument('--soca_stage', type=int, default=1,
                        help='SOCA analysis iteration (1==ombg, 2==oman)')
                        
    # Add DA cycle as an argument (optional, default to 6.0)
    parser.add_argument('--da_cycle', type=float, default=24.,
                        help='The DA cycle duration in hours (default: 6.0)')
    
    # Add days to smooth as an argument (optional, default to 8.0)
    parser.add_argument('--days_to_smooth', type=float, default=8.,
                        help='Number of days to smooth (default: 8.0)')
    parser.add_argument('--qc_threshold', type=int, default=0)
                        
    parser.add_argument(
        '--dark_theme', 
        action='store_true',  # If this argument is provided, dark_theme will be True
        help="Enable dark theme (default is False)"
    )
    
    args = parser.parse_args()

    return args

def get_data_frame(experiment_list, sensor_list, variable_list,
                   start_date='1978-10-01 12:00:00',
                   stop_date='2025-09-30 12:00:00',
                   region='global',
                   soca_it=1):
        
    soca_it = int(soca_it)
    if soca_it == 1:
        group = 'ombg'
    elif soca_it == 2:
        group = 'oman'
    
    metric_list = list()
    for sensor in sensor_list:
        for variable in variable_list:
            metric_list.append(f'mean_{variable}_{sensor}_{group}')
            metric_list.append(f'StdDev_{variable}_{sensor}_{group}')
            metric_list.append(f'mean_{variable}_{sensor}_ObsError')
            metric_list.append(f'rms_{variable}_{sensor}_{group}')
            metric_list.append(f'count_{variable}_{sensor}_{group}')
        
    return soca_diags_timeseries.get_data_frame(
                                                experiment_list,
                                                metric_list,
                                                start_date=start_date,
                                                stop_date=stop_date,
                                                region=region)

class SOCADiagsFit2ObsFig(object):
    """
    """
    def __init__(self, data_frame=None, input_data_frame=False,
                 soca_it=1):
        """
        """
        self.soca_it = int(soca_it)
        self.config_dict, self.friendly_names_dict = config()
        self.experiment_list = self.config_dict['experiment_list']
        self.sensor_list = self.config_dict['sensor_list']
        self.variable_list = self.config_dict['variable_list']
        self.region = self.config_dict['region']
                
        if input_data_frame:
            self.data_frame = data_frame
        else:
            self.data_frame = get_data_frame(self.experiment_list,
                                             self.config_dict['start_date'],
                                             self.config_dict['stop_date'],
                                             region=self.region,
                                             soca_it=self.soca_it)
    
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
    
    def build_timeseries(self, interactive_figure=False, ncols=3,
                         da_cycle = 6., # hours
                         days_to_smooth = 1., # days
                         dark_theme=False,
                         qc_threshold=0):
        self.dark_theme = dark_theme
        self.da_cycle = da_cycle
        self.config_figure_params(days_to_smooth=days_to_smooth)
        figsize_width = 4.5 * ncols
        
        #TODO: need to change figsize_length so that it only counts sensors
        # 
        figsize_length = 4.53 * len(self.sensor_list)
        
        
        output_dir = self.config_dict['output_path']
        locator = mdates.AutoDateLocator(minticks=8, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        
        month_locator = mdates.MonthLocator(interval=3)
        
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        nrows = 0
        
        if self.soca_it == 1:
            self.group = 'ombg'
        elif self.soca_it >= 2:
            self.group = 'oman'
        
        for variable in self.variable_list:
            
            # instantiate figure here    
            self.fig, self.axes = plt.subplots(len(self.sensor_list), ncols, sharex=True,sharey=False,
                                     squeeze=False,
                                     figsize=(figsize_width, figsize_length))
            
            axes_row = 0
            for sensor in sorted(self.sensor_list):
                self.max_yerr=0.
                
                if sensor in self.config_dict['sensor_list'] and variable in self.config_dict['variable_list']:
                    experiment_timeseries_datetime_init=None
                    metric_list = [f'mean_{variable}_{sensor}_{self.group}',
                                   f'StdDev_{variable}_{sensor}_{self.group}',
                                   f'mean_{variable}_{sensor}_ObsError',
                                   f'rms_{variable}_{sensor}_{self.group}',
                                   f'count_{variable}_{sensor}_{self.group}',
                                 ]
                                     
                    self.experiment_timeseries_dict = dict()
                    for experiment in self.experiment_list:
                        self.experiment_timeseries_dict[experiment] = dict()
                        for metric in metric_list:
                            try:
                                self.experiment_timeseries_dict[
                                    experiment][metric] = soca_diags_timeseries.SOCADiagsTimeSeries(
                                                    self.config_dict['start_date'],
                                                    self.config_dict['stop_date'],
                                                    data_frame=self.data_frame,
                                                    input_data_frame=True,
                                                    experiment_name=experiment,
                                                    metric_types=metric)
                                                    
                                experiment_timeseries_datetime_init = self.experiment_timeseries_dict[
                                    experiment][metric].init_datetime
                                
                                self.experiment_timeseries_dict[experiment][metric].build(qc_threshold=qc_threshold)
                                
                            except KeyError: # remove metric from dict if no records returned                
                                self.experiment_timeseries_dict[experiment].pop(metric, None)
                                warnings.warn(f'missing {sensor} {variable} records for '
                                              f'{experiment} experiment: {metric}')
                
                    self.db_name = os.getenv('SCORE_POSTGRESQL_DB_NAME')        
                    
                    # subplot titles
                    self.axes[axes_row, 0].set_title(f'Bias: {self.friendly_names_dict[sensor]}')
                    self.axes[axes_row, 1].set_title(f'RMSE: {self.friendly_names_dict[sensor]}')
                    self.axes[axes_row, 2].set_title(f'N obs: {self.friendly_names_dict[sensor]}')
        
                    # vertical axes labels
                    units = self.config_dict['units'][variable]
                    self.axes[axes_row, 0].set_ylabel(f'{variable} mean error ({units})')
                    self.axes[axes_row, 1].set_ylabel(f'{variable} RMS error ({units})')
                    self.axes[axes_row, 2].set_ylabel('Number of obs used')

                    if self.dark_theme:
                        self.axes[axes_row, 0].axhline(color='#A2A4A3', lw=0.5)
                    else:
                        self.axes[axes_row, 0].axhline(color='black', lw=0.5)

                    # Set ticks on both left and right vertical axes
                    self.axes[axes_row, 0].tick_params(axis='y', which='both', left=True, right=True)
                    self.axes[axes_row, 1].tick_params(axis='y', which='both', left=True, right=True)
                    self.axes[axes_row, 2].tick_params(axis='y', which='both', left=True, right=True)
        
                    self.axes[axes_row, 0].tick_params(axis='x', which='both', top=True, bottom=True,
                                             labelbottom=True)
                    self.axes[axes_row, 1].tick_params(axis='x', which='both', top=True, bottom=True,
                                             labelbottom=True)
                    self.axes[axes_row, 2].tick_params(axis='x', which='both', top=True, bottom=True,
                                             labelbottom=True)
                    
                    self.plot_data(axes_row, variable, sensor)
                    
                    for col_idx in range(ncols):
                        if self.dark_theme:
                            self.axes[axes_row, col_idx].grid(True)
                        self.axes[axes_row, col_idx].xaxis.set_major_locator(locator)
                        self.axes[axes_row, col_idx].xaxis.set_major_formatter(formatter)
                        self.axes[axes_row, col_idx].xaxis.set_minor_locator(month_locator)
                    
                    axes_row += 1
            
            '''
            for row, sensor in enumerate(sorted(self.sensor_list)):
                
                # set ylim, ticks
                if max_yerr < 1:
                    axes[row, 0].set_yticks(
                        np.arange(np.around(-0.5 * max_yerr - 0.2, decimals=1),
                                  0.5 * max_yerr + 0.2,
                                  0.1),
                    )
                else:
                    axes[row, 0].set_yticks(
                        np.arange(np.around(-0.5 * max_yerr - 1),
                                  0.5 * max_yerr + 1,
                                  0.5),
                    )
                    axes[row, 0].set_yticks(
                        np.arange(np.around(-0.5 * max_yerr - 0.2, decimals=1),
                                  0.5 * max_yerr + 0.2,
                                  0.1),
                                  minor=True
                    )
            
                axes[row, 1].set_yticks(
                    np.arange(0, 3. * max_yerr + 0.1, 0.1),
                    minor=True
                    )
            
                axes[row, 1].set_yticks(
                    np.arange(0, 3. * max_yerr + 1, 0.5)
                )
            
                        axes[row, 0].set_ylim(-0.5*max_yerr, 0.5*max_yerr)
                        axes[row, 1].set_ylim(0, 3.*max_yerr)
            
                        nobs_ylims = axes[row, 2].get_ylim()
                        if nobs_ylims[0] < 0:
                            axes[row, 2].set_ylim(bottom=0)
                            
                '''            
        
            if self.soca_it == 1:
                title_str0 = f"GDAS {self.region} background fit to observations (O-B) [metrics downloaded from {self.db_name}"
            elif self.soca_it >= 2:
                title_str0 = f"GDAS {self.region} analysis fit to observations (O-A) [metrics downloaded from {self.db_name}"
            
            if experiment_timeseries_datetime_init:
                init_ctime = experiment_timeseries_datetime_init.ctime()
                title_str1 = f" {init_ctime}]"
            else:
                title_str1 = "]"
            
            self.fig.suptitle(f"{title_str0}{title_str1}")
            plt.tight_layout()
            plt.subplots_adjust(top = 1. - 1.2 / figsize_length)

            if self.region=='global':
                if self.soca_it ==1:
                    fig_title=f'gdas_ocean_{variable}_omb.png'
                elif self.soca_it >=2:
                    fig_title=f'gdas_ocean_{variable}_oma.png'
                    
            else:
                if self.soca_it ==1:
                    fig_title=f'gdas_{self.region}_{variable}_omb.png'
                elif self.soca_it >=2:
                    fig_title=f'gdas_{self.region}_{variable}_oma.png'
            
            if interactive_figure:
                plt.show()
            else:
                plt.savefig(os.path.join(output_dir, fig_title), dpi=300)
            plt.close()
    
    def plot_data(self, axes_row, variable, sensor,
                     alpha_foreground=0.9,
                     alpha_background=0.5,):    
        
        #experiment_idx = 0
        for experiment, timeseries_dict in self.experiment_timeseries_dict.items():
            for full_stat_name, timeseries_data in timeseries_dict.items():
                for stat_label, value_dict in timeseries_data.value_dict.items():
                    
                    if stat_label == f'mean_{variable}_{self.group}' and sensor in timeseries_data.timestamp_dict[stat_label].keys():
                        """ mean error plot
                        """
                        bias_timeseries = pd.Series(
                            data=np.array(
                                value_dict[sensor]
                            ),
                            index=timeseries_data.timestamp_dict[stat_label][sensor]
                        ).astype(float)
                        
                        std_timeseries = pd.Series(
                            data=np.array(
                                timeseries_dict[f'StdDev_{variable}_{sensor}_{self.group}'].value_dict[
                                    f'StdDev_{variable}_{self.group}'][sensor]
                            ),
                            index=timeseries_dict[f'StdDev_{variable}_{sensor}_{self.group}'].timestamp_dict[
                                f'StdDev_{variable}_{self.group}'][sensor]
                        ).astype(float)
                        
                        nobs_used_timeseries = pd.Series(
                            data=np.array(
                                timeseries_dict[f'count_{variable}_{sensor}_{self.group}'].value_dict[
                                    f'count_{variable}_{self.group}'][sensor]
                            ),
                            index=timeseries_dict[f'count_{variable}_{sensor}_{self.group}'].timestamp_dict[
                                f'count_{variable}_{self.group}'][sensor]
                        ).astype(float)

                        standard_errs = std_timeseries / np.sqrt(nobs_used_timeseries)

                        bias_timeseries = bias_timeseries.combine_first(
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
                        
                        self.axes[axes_row, 0].fill_between(
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
                        
                        self.axes[axes_row, 0].plot(
                            mean_values_smooth.index,
                            mean_values_smooth.values,
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

                        self.axes[axes_row,0].legend(loc='upper right')
                
                    elif stat_label == f'rms_{variable}_{self.group}' and sensor in timeseries_data.timestamp_dict[stat_label].keys():
                        """RMS error plot
                        """
                        rmse_timeseries = pd.Series(
                            data=np.array(
                                value_dict[sensor]
                            ),
                            index=timeseries_data.timestamp_dict[stat_label][sensor]
                        ).astype(float)
                        
                        mean_obs_err_timeseries = pd.Series(
                            data=np.array(
                                timeseries_dict[f'mean_{variable}_{sensor}_ObsError'].value_dict[
                                    f'mean_{variable}_ObsError'][sensor]
                            ),
                            index=timeseries_dict[f'mean_{variable}_{sensor}_ObsError'].timestamp_dict[
                                f'mean_{variable}_ObsError'][sensor]
                        ).astype(float)

                        self.max_yerr = np.max(
                            np.nan_to_num(mean_obs_err_timeseries.values),
                            initial=self.max_yerr
                        )
                        
                        rmse_timeseries = rmse_timeseries.combine_first(
                            self.time_domain
                        )
                        
                        rmse_values_smooth = rmse_timeseries.rolling(
                            window=self.window_size,
                            min_periods=self.min_periods,
                            center=True,
                            #win_type='triang'
                        ).mean()
                    
                        self.axes[axes_row, 1].plot(
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
                        
                        self.axes[axes_row, 1].plot(
                            rmse_values_smooth.index,
                            rmse_values_smooth.values,
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
                                        
                    elif stat_label == f'count_{variable}_{self.group}' and sensor in timeseries_data.timestamp_dict[stat_label].keys():
                        nobs_used_timeseries = pd.Series(
                            data=np.array(
                                value_dict[sensor]
                            ),
                            index=timeseries_data.timestamp_dict[stat_label][sensor]
                        ).astype(float)

                        nobs_used_timeseries = nobs_used_timeseries.combine_first(
                            self.time_domain
                        )
                        
                        nobs_used_smooth = nobs_used_timeseries.rolling(
                            window=self.window_size,
                            min_periods=self.min_periods,
                            center=True,
                            #win_type='triang'
                        ).mean()
                        
                        self.axes[axes_row, 2].plot(
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
                        
                        self.axes[axes_row, 2].plot(
                            nobs_used_smooth.index,
                            nobs_used_smooth.values,
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

def prun(experiment_list=None, sensor_list=None, variable_list=None, start_date=None, stop_date=None, region=None):
    args = parse_arguments()
    soca_stage = args.soca_stage
    
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
            
    if region is None:
        region = global_config_dict['region']
    
    if args.variable != 'all':
        variable_list = [args.variable]
    
    if variable_list is None:
        variable_list = list()
        for var in global_config_dict['variable_list']:
            variable_list.append(var)
            
    # Rank 0 prepares the data
    if rank == 0:
        global_data_frame = get_data_frame(
            experiment_list,
            sensor_list,
            variable_list,
            region=region,
            start_date=start_date,
            stop_date=stop_date,
            soca_it=soca_stage)

        # Split the data by variable (one part per variable)
        data_frame_parts_dict = dict()

        for var in variable_list:
            data_frame_parts_dict[var] = global_data_frame[
                global_data_frame['metric_type'] == var]

    else:
        data_frame_parts_dict = None

    # Calculate how many variables each process should handle
    vars_per_process = len(variable_list) // size

    # Handle leftover sensors (remaining variables are distributed to the first few processes)
    leftover_vars = len(variable_list) % size
    
    if rank == 0:
        for i in range(0, size):
            # Calculate the subset of variables for this rank
            start_idx = i * vars_per_process + min(i, leftover_vars)
            end_idx = start_idx + vars_per_process + (1 if i < leftover_vars else 0)
            rank_vars = variable_list[start_idx:end_idx]
            
            # Prepare the data for this rank
            data_to_send = {var: data_frame_parts_dict[var] for var in rank_vars}

            if i==0:
                local_data_frames = data_to_send
            else:
                comm.send(data_to_send, dest=i, tag=11+i)
    else:
        local_data_frames = comm.recv(source=0, tag=11+rank)

    # Each process works on its part of the data
    if local_data_frames is not None:
        for var, data_frame in local_data_frames.items():
            # Only work on data for the specific variable assigned to the process
            '''
            print(f"Rank {rank} is processing variable: {var} and here is "
                   f"the data frame: {data_frame.metric_type}")
            
            '''
            experiment_metrics_timeseries_data = SOCADiagsFit2ObsFig(
                data_frame=data_frame,
                input_data_frame=True,
                soca_it=soca_stage
            )
            experiment_metrics_timeseries_data.variable_list = [var]
            experiment_metrics_timeseries_data.sensor_list = sensor_list
            experiment_metrics_timeseries_data.experiment_list = experiment_list
            experiment_metrics_timeseries_data.region = region
            experiment_metrics_timeseries_data.config_dict['start_date'] = start_date
            experiment_metrics_timeseries_data.config_dict['stop_date'] = stop_date
            experiment_metrics_timeseries_data.build_timeseries(interactive_figure=args.interactive,
                                                                days_to_smooth=args.days_to_smooth,
                                                                da_cycle=args.da_cycle,
                                                                dark_theme=args.dark_theme,
                                                                qc_threshold=args.qc_threshold)

def main():
    """
    """
    prun(sensor_list=['avhrr', 'viirs'],
         variable_list=['seaSurfaceTemperature'])
    prun(sensor_list=['amsr2', 'ssmis'],
        variable_list=['seaIceFraction'],region='nh')
    prun(sensor_list=['amsr2', 'ssmis'],
        variable_list=['seaIceFraction'],region='sh')

if __name__ == "__main__":
    main()
