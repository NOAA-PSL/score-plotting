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

from score_plotting.core_scripts import gsistats_conv_timeseries

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
            #'NASA_GEOSIT_GSISTATS',
            #'GDAS',
            #'replay_observer_diagnostic_v1',
            'scout_run_v1',
            '3dvar_coupledreanl_scoutrun_1979streamv1_test1'
                         ],
        
        'experiment_plot_dict': {
                        'NASA_GEOSIT_GSISTATS' :
                {'color' : '#E4002B',
                 'ls': '-',
                 'lw': 1.5
            },
            'GDAS' : {
                'color' : '#003087',
                'ls': '-',
                'lw': 1.25
            },
            'replay_observer_diagnostic_v1' : {
                'color' : '#0085CA',
                'ls': '-',
                'lw': 1.
            },
            'scout_run_v1' : {
                'color' : 'black',
                'ls': '-',
                'lw': 1.5
            },
            'replay_observer_diagnostic_overlap' : {
                'color' : 'black',
                'ls': '-',
                'lw': 0.75
            },
            '3dvar_coupledreanl_scoutrun_1979streamv1_test1' : {
                'color' : '#003087',
                'ls': '-',
                'lw': 1.25
            }
        },
        'sensor_list': [111, 112, 120, 122, 126, 130, 131, 132, 133, 134, 135,
                        150, 151, 152, 153, 154, 156, 157, 158, 159, 164, 165,
                        170, 171, 174, 175, 180, 181, 182, 183, 187, 188, 191,
                        192, 193, 194, 195, 210, 220, 221, 222, 223, 224, 227,
                        228, 229, 230, 231, 232, 233, 234, 235, 240, 241, 242,
                        243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
                        254, 255, 256, 257, 258, 259, 260, 270, 271, 280, 281,
                        282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
                        293, 294, 295, 'all'
                    ]  ,
        'variable_list': [
            'fit_psfc_data', # fit of surface pressure data (hPa)
            'fit_uv_data', # fit of u, v wind data (m/s)
            'fit_t_data', # fit of temperature data (K)
            'fit_q_data', # fit of moisture data (% of qsaturation guess)
        ],
        'start_date': '1978-10-01 00:00:00',
        'stop_date': '2025-09-30 23:59:59',
    }
    
    '''
    could this be done by string matching for the std/bias etc part? we could
    have a basic friendly dict for that
    '''
    friendly_names_dict={
            "scout_run_v1": "atmosphere scout (3DVar)",
            "NASA_GEOSIT_GSISTATS": "GEOS-IT",
            "GDAS": "GDAS",
            "replay_observer_diagnostic_v1": "UFS-replay",
            "replay_observer_diagnostic_overlap": "UFS-replay-overlap",
            "3dvar_coupledreanl_scoutrun_1979streamv1_test1": "weakly coupled scout (3DVar)",
                         }
                         
    return(config_dict, friendly_names_dict)
    
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Script to create GSI diagnostics timeseries figures for '
                    'conventional surface observation monitoring')
    
    # Make figure_output_path optional (defaults to $HOME)
    parser.add_argument('figure_output_path', type=str, nargs='?',
                        default=pathlib.Path.home(),
                        help='Path to where figures will be saved')
    
    # Add an argument for interactive plotting
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive plotting. If specified, plots will be displayed interactively.')
    
    # Add optional argument for sensor
    parser.add_argument('--sensor', type=str, default='all',
                        help='Sensor name (e.g., rawinsonde)')
                        
    # Add optional argument for variable
    parser.add_argument('--variable', type=str, default='all',
                        help='variable name (e.g., fit_psfc_data)')
                        
    parser.add_argument('--gsi_stage', type=int, default=1,
                        help='GSI analysis iteration (1==ombg, 2==oman)')
                        
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
    
    args = parser.parse_args()

    return args

def get_data_frame(experiment_list, sensor_list, variable_list,
                   start_date='1978-10-01 00:00:00',
                   stop_date='2025-09-30 23:59:59',
                   gsi_it=1):
        
    gsi_it = int(gsi_it)
    
    metric_list = list()
    for sensor_id in sensor_list:
        for variable in variable_list:
            metric_list.append(f'count_{variable}_{sensor_id}_GSIstage_{gsi_it}')
            metric_list.append(f'bias_{variable}_{sensor_id}_GSIstage_{gsi_it}')
            metric_list.append(f'rms_{variable}_{sensor_id}_GSIstage_{gsi_it}')
        
    return gsistats_conv_timeseries.get_data_frame(
                                                experiment_list,
                                                metric_list,
                                                start_date=start_date,
                                                stop_date=stop_date)

class GSIConvFit2ObsFig(object):
    """
    """
    def __init__(self, data_frame=None, input_data_frame=False,
                 gsi_it=1):
        """
        """
        self.gsi_it = int(gsi_it)
        self.config_dict, self.friendly_names_dict = config()
        self.experiment_list = self.config_dict['experiment_list']
        self.sensor_list = self.config_dict['sensor_list']
        self.variable_list = self.config_dict['variable_list']
                
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
    
    def build_timeseries(self, interactive_figure=False, ncols=3,
                         da_cycle = 6., # hours
                         days_to_smooth = 1., # days
                         dark_theme=False):
        self.dark_theme = dark_theme
        self.da_cycle = da_cycle
        self.config_figure_params(days_to_smooth=days_to_smooth)
        figsize_width = 2 * 3.74 * ncols
            
        output_dir = self.config_dict['output_path']
        locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
        formatter = mdates.ConciseDateFormatter(locator)
        
        month_locator = mdates.MonthLocator(interval=1)
        
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for variable in self.variable_list:
            local_data_frame = self.data_frame[self.data_frame['name'].str.contains(variable)]
            sensors_to_show = set(local_data_frame.metric_instrument_name)
            nrows = len(sensors_to_show)        
            figsize_length = 4.53 * nrows
            # instantiate figure here    
            self.fig, self.axes = plt.subplots(nrows, ncols, sharex=True,sharey=False,
                                     squeeze=False,
                                     figsize=(figsize_width, figsize_length))
            
            axes_row = 0
            for sensor in sensors_to_show:
    
                self.max_yerr=0.
                if variable in self.variable_list and sensor is not None:
                    experiment_timeseries_datetime_init=None
                    
                    data_frame_to_show = local_data_frame[local_data_frame.metric_instrument_name==sensor]
                    
                    metric_list = set(data_frame_to_show.name)
                                     
                    self.experiment_timeseries_dict = dict()
                    for experiment in self.experiment_list:
                        self.experiment_timeseries_dict[experiment] = dict()
                        for metric in metric_list:
                            try:
                                self.experiment_timeseries_dict[
                                    experiment][metric] = gsistats_conv_timeseries.GSIConvTimeSeries(
                                                    self.config_dict['start_date'],
                                                    self.config_dict['stop_date'],
                                                    data_frame=data_frame_to_show,
                                                    input_data_frame=True,
                                                    experiment_name=experiment,
                                                    metric_types=metric)
                                                                                                        
                                experiment_timeseries_datetime_init = self.experiment_timeseries_dict[
                                    experiment][metric].init_datetime
                            
                                self.experiment_timeseries_dict[experiment][metric].build()
                                
                            except KeyError: # remove metric from dict if no records returned                
                                self.experiment_timeseries_dict[experiment].pop(metric, None)
                                warnings.warn(f'missing {sensor} {variable} records for '
                                              f'{experiment} experiment: {metric}')
                
                    self.db_name = os.getenv('SCORE_POSTGRESQL_DB_NAME')        
                    
                    # subplot titles
                    self.axes[axes_row, 0].set_title(f'Bias: {sensor} ({data_frame_to_show.metric_obs_platform.values[0]})')
                    self.axes[axes_row, 1].set_title(f'RMS: {sensor} ({data_frame_to_show.metric_obs_platform.values[0]})')
                    self.axes[axes_row, 2].set_title(f'Nobs assimilated: {sensor} ({data_frame_to_show.metric_obs_platform.values[0]})')
        
                    # vertical axes labels
                    self.axes[axes_row, 0].set_ylabel(f'Bias {data_frame_to_show.metric_long_name.values[0]}')
                    self.axes[axes_row, 1].set_ylabel(f'RMS {data_frame_to_show.metric_long_name.values[0]}')
                    self.axes[axes_row, 2].set_ylabel('Number of obs assimilated')

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
        
                if self.gsi_it == 1:
                    difference_str = "Ob - Bg"
                elif self.gsi_it >= 2:
                    difference_str = "Ob - Anal"
                
                title_str0 = f"GSI conventional data anal fit to assimilated obs ({difference_str}) [metrics downloaded: {self.db_name}"
            
            if experiment_timeseries_datetime_init:
                init_ctime = experiment_timeseries_datetime_init.ctime()
                title_str1 = f" {init_ctime}]"
            else:
                title_str1 = "]"
            
            self.fig.suptitle(f"{title_str0}{title_str1}")
            plt.tight_layout()
            plt.subplots_adjust(top = 1. - 1.2 / figsize_length)
            if interactive_figure:
                plt.show()
            else:
                if self.gsi_it ==1:
                    fig_title=f'gdas_gsi_conv_asm_omb_{variable}.png'
                elif self.soca_it >=2:
                    fig_title=f'gdas_gsi_conv_asm_oma_{variable}.png'
                plt.savefig(os.path.join(output_dir, fig_title), dpi=300)
            plt.close()
    
    def plot_data(self, axes_row, variable, sensor,
                     alpha_foreground=0.9,
                     alpha_background=0.5,):    
        
        #experiment_idx = 0
        for experiment, timeseries_dict in self.experiment_timeseries_dict.items():
            for metric, timeseries_data in timeseries_dict.items():
                if sensor in timeseries_data.timestamp_dict[metric].keys():
                    value_arr = timeseries_data.value_dict[metric][sensor]['asm']
                    timestamp_arr = timeseries_data.timestamp_dict[metric][sensor]['asm']
                if metric.split('_')[0] == 'bias' and sensor in timeseries_data.timestamp_dict[metric].keys() and len(timestamp_arr) > 0:
                    """ mean error plot
                    """
                    bias_timeseries = pd.Series(
                        data=np.array(
                            value_arr
                        ),
                        index=timestamp_arr
                    ).astype(float)

                    bias_timeseries = bias_timeseries.combine_first(
                        self.time_domain
                    )
                        
                    mean_values_smooth = bias_timeseries.rolling(
                        window=self.window_size,
                        min_periods=self.min_periods,
                        center=True,
                    ).mean()
                        
                    self.axes[axes_row, 0].plot(
                        bias_timeseries.index,
                        bias_timeseries.values,
                        marker='none',
                        color=self.config_dict['experiment_plot_dict']
                            [experiment]['color'],
                        alpha=alpha_background,
                        lw=0.5,
                        ls='-',
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
                
                elif metric.split('_')[0] == 'rms' and sensor in timeseries_data.timestamp_dict[metric].keys() and len(timestamp_arr) > 0:
                    """RMS error plot
                    """
                    rmse_timeseries = pd.Series(
                        data=np.array(
                            value_arr
                        ),
                        index=timestamp_arr
                    ).astype(float)
                            
                    rmse_timeseries = rmse_timeseries.combine_first(
                            self.time_domain
                    )
                        
                    rmse_values_smooth = rmse_timeseries.rolling(
                        window=self.window_size,
                        min_periods=self.min_periods,
                        center=True,
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
                                        
                elif metric.split('_')[0] == 'count' and sensor in timeseries_data.timestamp_dict[metric].keys() and len(timestamp_arr) > 0:
                    nobs_used_timeseries = pd.Series(
                        data=np.array(
                            value_arr
                        ),
                        index=timestamp_arr
                    ).astype(float)

                    nobs_used_timeseries = nobs_used_timeseries.combine_first(
                        self.time_domain
                    )
                        
                    nobs_used_smooth = nobs_used_timeseries.rolling(
                        window=self.window_size,
                        min_periods=self.min_periods,
                        center=True,
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

def prun(experiment_list=None, sensor_list=None, variable_list=None, start_date=None, stop_date=None):
    args = parse_arguments()
    gsi_stage = args.gsi_stage
    
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
            start_date=start_date,
            stop_date=stop_date,
            gsi_it=gsi_stage)

        # Split the data by variable (one part per variable)
        data_frame_parts_dict = dict()
        
        for var in variable_list:
            data_frame_parts_dict[var] = global_data_frame[
                global_data_frame['name'].str.contains(var)]

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
            experiment_metrics_timeseries_data = GSIConvFit2ObsFig(
                data_frame=data_frame,
                input_data_frame=True,
                gsi_it=gsi_stage
            )
            experiment_metrics_timeseries_data.variable_list = [var]
            experiment_metrics_timeseries_data.config_dict['sensor_list'] = sensor_list
            experiment_metrics_timeseries_data.experiment_list = experiment_list
            experiment_metrics_timeseries_data.config_dict['start_date'] = start_date
            experiment_metrics_timeseries_data.config_dict['stop_date'] = stop_date
            experiment_metrics_timeseries_data.build_timeseries(interactive_figure=args.interactive,
                                                                days_to_smooth=args.days_to_smooth,
                                                                da_cycle=args.da_cycle,
                                                                dark_theme=args.dark_theme)

def main():
    """
    """
    prun()

if __name__ == "__main__":
    main()
