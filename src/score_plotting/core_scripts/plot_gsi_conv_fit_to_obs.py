#!/usr/bin/env python

"""
"""

import os
import pathlib
import warnings
import argparse
from datetime import datetime

import numpy as np
from scipy.signal import detrend
from scipy import stats as scipy_stats
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import colorcet as cc
import pandas as pd
from mpi4py import MPI

from score_plotting.core_scripts import gsistats_conv_timeseries

HOURS_PER_DAY = 24. # hours
P_TEST_LEVEL = 0.01

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
        'experiment_list': [
            'cfsr',
            'GDAS',
            'NASA_GEOSIT_GSISTATS',
            'replay_observer_diagnostic_v1.1',
            'scout_run_v1',
            '3dvar_coupledreanl_scoutrun_v1_test1',
            '3dvar_coupledreanl_scoutrun_v2'
                         ],
        
        'experiment_plot_dict': {
                        'NASA_GEOSIT_GSISTATS' :
                {'color' : '#E4002B',
                 'color2': '#A2A4A3',
                 'marker': 'x',
                 'ls': '-',
                 'ls2': ':',
                 'lw': 1.0,
                 'zorder':1
            },
            'cfsr' :
               {'color' : 'green',
                'color2': '#A2A4A3',
                'marker': 'x',
                'ls': '-',
                'ls2': ':',
                'lw': 1.5,
                'zorder':1,
            },
            
            'GDAS' : {
                'color' : '#0085CA',
                'ls': '-',
                'lw': 1.25
            },
            'replay_observer_diagnostic_v1.1' : {
                'color' : '#096FAE',#'#0A3758',
                'color2': '#EEF5F8',
                'marker': '+',
                'ls': '-',
                'lw': 0.75,
                'ls2': '-.',
                'zorder':2
            },
            'scout_run_v1' : {
                'color' : '#000000',
                'ls': '-',
                'lw': 0.5
            },
            'replay_observer_diagnostic_overlap' : {
                'color' : '#8D7334',
                'ls': '-',
                'lw': 0.75
            },
            '3dvar_coupledreanl_scoutrun_v1_test1' : {
                'color' : '#565A5C',
                'ls': '-',
                'lw': 0.5
            },
            '3dvar_coupledreanl_scoutrun_v2' : {
                'color' : '#8D7334',
                'color2': '#F3F0E9',
                'marker': '+',
                'ls': '-',
                'lw': 0.75,
                'ls2': '--',
                'zorder':2
            }
        },
        'sensor_list': [111, 112, 120, 122, 126, 130, 131, 132, 133, 134, 135,
                        150, 151, 152, 153, 154, 156, 157, 158, 159, 164, 165,
                        170, 171, 174, 175, 180, 181, 182, 183, 187, 188, 191,
                        192, 193, 194, 195, 199, 210, 220, 221, 222, 223, 224, 227,
                        228, 229, 230, 231, 232, 233, 234, 235, 240, 241, 242,
                        243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
                        254, 255, 256, 257, 258, 259, 260, 270, 271, 280, 281,
                        282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
                        293, 294, 295, 299, 'all'
                    ]  ,
        'variable_list': [
            'fit_psfc_data', # fit of surface pressure data (hPa)
            #'fit_uv_data', # fit of u, v wind data (m/s)
            #'fit_t_data', # fit of temperature data (K)
            #'fit_q_data', # fit of moisture data (% of qsaturation guess)
        ],
        'start_date': '1996-10-01 00:00:00',
        'stop_date': '2026-09-30 23:59:59',
    }
    
    '''
    could this be done by string matching for the std/bias etc part? we could
    have a basic friendly dict for that
    '''
    friendly_names_dict={
            "scout_run_v1": "scout-atm (v0.01)",
            "NASA_GEOSIT_GSISTATS": "GEOS-IT",
            "GDAS": "GDAS",
            "replay_observer_diagnostic_v1.1": "Replay",
            "replay_observer_diagnostic_overlap": "UFS-replay-overlap",
            "3dvar_coupledreanl_scoutrun_v1_test1": "scout-1 (v0.1)",
            '3dvar_coupledreanl_scoutrun_v2': "scout-2 (v0.21)",
            'cfsr':"CFSR"
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
                        
    parser.add_argument('--pressure_bins', action='store_true',help="analyze array metrics by pressure bins")
                        
    parser.add_argument(
        '--dark_theme', 
        action='store_true',  # If this argument is provided, dark_theme will be True
        help="Enable dark theme (default is False)"
    )
    
    args = parser.parse_args()

    return args

def get_data_frame(experiment_list, sensor_list, variable_list,
                   start_date='1978-10-01 00:00:00',
                   stop_date='2026-09-30 23:59:59',
                   gsi_it=1,
                   pressure_bins=False):
        
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
                                                stop_date=stop_date,
                                                array=pressure_bins)

def lag1_autocorrelation(timeseries):
    """
    """
    detrended_timeseries = detrend(timeseries)
    
    timeseries_deviation_minus = detrended_timeseries[:-1] - np.mean(detrended_timeseries[:-1], axis=0)
    timeseries_deviation_plus = detrended_timeseries[1:] - np.mean(detrended_timeseries[1:], axis=0)
    
    return (np.sum((timeseries_deviation_minus) * (timeseries_deviation_plus), axis=0)
        ) / (np.sqrt(np.sum(timeseries_deviation_minus**2, axis=0)) * np.sqrt(np.sum(timeseries_deviation_plus**2, axis=0)))
        
def variance_time_avg(stdev, n, row1=0.):
    """
    """
    variance_inflation_factor = (1. + row1) / (1. - row1)
    variance = (stdev**2 / n) * variance_inflation_factor
    
    n_eff = n * variance_inflation_factor**-1
    
    return(variance, n_eff)
    
def get_tstat(difference_timeseries, null=0.):
    
    try:
        variance, n_eff = variance_time_avg(
                              difference_timeseries.std(axis=0, ddof=1),
                              difference_timeseries.count(axis=0),
                              row1=lag1_autocorrelation(difference_timeseries))
    
        tstat = (difference_timeseries.mean(axis=0) - null) / np.sqrt(variance)
        p_value = 2. * scipy_stats.t.sf(abs(tstat), df=n_eff - 1.0)
    except ValueError:
        warnings.warn(f'lag1_autocorrelation({difference_timeseries}) cannot be computed, possibly due to the existance of infs or NaNs')
        tstat=None
        p_value=None
    
    return(tstat, p_value)

class GSIConvFit2ObsFig(object):
    """
    """
    def __init__(self, data_frame=None, input_data_frame=False,
                 gsi_it=1, pressure_bins=False):
        """
        """
        self.gsi_it = int(gsi_it)
        self.array = pressure_bins
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
                                             gsi_it=self.gsi_it,
                                             pressure_bins=self.array)
    
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
        
        if self.array:
            self.time_domain = pd.DataFrame(
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
            
            self.time_domain_bnds = pd.DataFrame(
                    index = pd.date_range(
                        start = datetime.strptime(
                            self.config_dict['start_date'], '%Y-%m-%d %H:%M:%S'
                        ) - pd.Timedelta(hours = self.da_cycle/2.),
                        end = datetime.strptime(
                            self.config_dict['stop_date'], '%Y-%m-%d %H:%M:%S'
                        ) + pd.Timedelta(hours = self.da_cycle/2.),
                        freq = pd.Timedelta(hours = self.da_cycle)
                    )
                )
        else:
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
    
    def build_timeseries(self, interactive_figure=False, ncols=3, #ncols is number of metrics (e.g., bias, rmse, nobs)
                         da_cycle = 6., # hours
                         days_to_smooth = 1., # days
                         dark_theme=False):
        if self.array:
            metric_name_key = 'metric_name'
        else:
            metric_name_key = 'name'
        
        self.interactive_figure = interactive_figure
        self.dark_theme = dark_theme
        self.da_cycle = da_cycle
        self.config_figure_params(days_to_smooth=days_to_smooth)
            
        output_dir = self.config_dict['output_path']
        
        self.locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
        self.formatter = mdates.ConciseDateFormatter(self.locator)
        self.month_locator = mdates.MonthLocator(interval=3)
        
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        
        if self.array:
            axes_row = 0
            iterator = (len(self.experiment_list) * (len(self.experiment_list)  + 1)) // 2
            nrows = iterator * ncols
            nmetrics=ncols
            ncols = 1
            figsize_length = 9.06 * (nrows/9.)
            figsize_width = 6.5#7.48
        
        
        for variable in self.variable_list:
            local_data_frame = self.data_frame[self.data_frame[metric_name_key].str.contains(variable)]
            
            sensors_to_show = sorted(set(local_data_frame.metric_instrument_name))
            
            if len(sensors_to_show) > 0:
                axes_row = 0
                if not self.array:
                    nrows = len(sensors_to_show)
                    iterator = 1
                    figsize_length = 4.53 * nrows
                    figsize_width = 4.5 * ncols
                    # instantiate figure here for single level data (line plots)    
                    self.fig, self.axes = plt.subplots(nrows, ncols, sharex=True,sharey=False,
                                            squeeze=False,
                                            figsize=(figsize_width, figsize_length))
            
                for sensor in sensors_to_show:
                    
                    if self.array:
                        axes_row=0
                    
                    self.max_yerr=0.
                    if variable in self.variable_list and sensor is not None:
                        experiment_timeseries_datetime_init=None
                    
                        data_frame_to_show = local_data_frame[local_data_frame.metric_instrument_name==sensor]
                    
                        metric_list = set(data_frame_to_show[metric_name_key])
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
                                                        metric_types=metric,
                                                        array=self.array)
                                                                                                        
                                    experiment_timeseries_datetime_init = self.experiment_timeseries_dict[
                                        experiment][metric].init_datetime
                            
                                    self.experiment_timeseries_dict[experiment][metric].build()
                                
                                except KeyError: # remove metric from dict if no records returned                
                                    self.experiment_timeseries_dict[experiment].pop(metric, None)
                                    warnings.warn(f'missing {sensor} {variable} records for '
                                                  f'{experiment} experiment: {metric}')
                
                        self.db_name = os.getenv('SCORE_POSTGRESQL_DB_NAME')
                        
                        if self.array:
                            self.seasonal_bias_xmax = 0
                            self.seasonal_rms_xmax = 0
                            self.seasonal_nobs_xmax = 0
                            # instantiate figure here for multi-level data (pcolormesh)
                            self.fig, self.axes = plt.subplots(nrows, ncols, sharex=True,sharey=True,
                                                    squeeze=False,
                                                    figsize=(figsize_width, figsize_length))
                        
                        self.plot_data(axes_row, variable, sensor, metric_unit=data_frame_to_show.metric_unit.values[0],
                                       iterator=iterator)
                        
                        self.config_fig(ncols, axes_row, sensor, data_frame_to_show, iterator=iterator)
        
                        if self.array:
                            self.finalize_fig(variable, sensor, data_frame_to_show,
                                          experiment_timeseries_datetime_init=experiment_timeseries_datetime_init,
                                          figsize_length=figsize_length, output_dir=output_dir)
                                          
                            self.fig, self.axes = plt.subplots(4, nmetrics, sharex=False, sharey=True,
                                                               squeeze=False, figsize=(figsize_width, figsize_length*(nmetrics/iterator)))
                            self.plot_data(axes_row, variable, sensor, metric_unit=data_frame_to_show.metric_unit.values[0],
                                           do_seasons=True)
                                           
                            self.finalize_fig(variable, sensor, data_frame_to_show,
                                          experiment_timeseries_datetime_init=experiment_timeseries_datetime_init,
                                          figsize_length=figsize_length*(nmetrics/iterator), output_dir=output_dir,
                                          do_seasons=True)
                        
                        axes_row += (1 * iterator)
                                          
                if not self.array:
                    self.finalize_fig(variable, sensor, data_frame_to_show,
                                      experiment_timeseries_datetime_init=experiment_timeseries_datetime_init,
                                      figsize_length=figsize_length, output_dir=output_dir)
                
    def finalize_fig(self, variable, sensor, data_frame_to_show, experiment_timeseries_datetime_init=False, figsize_length=9.06, output_dir=None,
                     do_seasons=False):
        if self.gsi_it == 1:
            difference_str = "Ob - Bg"
        elif self.gsi_it >= 2:
            difference_str = "Ob - Anal"
    
        if self.array:
            title_str0 = f"GSI conventional data anal fit to assimilated obs ({difference_str})\nmetrics downloaded: {self.db_name}"
        else:
            title_str0 = f"GSI conventional data anal fit to assimilated obs ({difference_str}) metrics downloaded: {self.db_name}"
        
        if experiment_timeseries_datetime_init:
            init_ctime = experiment_timeseries_datetime_init.ctime()
            title_str1 = f" {init_ctime}"
        else:
            title_str1 = ""

        if self.array:
            title_str1+=f"\n[{self.time_domain.index[0].strftime('%m-%d-%Y')} : {self.time_domain.index[-1].strftime('%m-%d-%Y')}] {data_frame_to_show.metric_long_name.values[0]}: {sensor}, {data_frame_to_show.metric_obs_platform.values[0]}"
            if do_seasons:
                title_str1+=f"\nStippling indicates $p$ < {P_TEST_LEVEL:.1g} (paired $t$-test; linear detrending; lag-1 autocorrelation adjustment)"
            
        self.fig.suptitle(f"{title_str0}{title_str1}")
        plt.tight_layout()
        plt.subplots_adjust(top = 1. - 1.2 / figsize_length)
        
        if self.interactive_figure:
            plt.show()
        else:
            if self.gsi_it ==1:
                fig_title=f'gdas_gsi_conv_asm_{variable}_omb.png'
            elif self.gsi_it >=2:
                fig_title=f'gdas_gsi_conv_asm_{variable}_oma.png'
            if self.array:
                str_sensor = str(self.sensor_id)
                fig_title = f'type_{str_sensor}_' + fig_title
            if do_seasons:
                fig_title = 'seasonal_' + fig_title
            plt.savefig(os.path.join(output_dir, fig_title), dpi=600)
        plt.close()                    
    
    def config_fig(self, ncols, axes_row, sensor, data_frame_to_show, iterator=None, nmetrics=3):
        # subplot titles
        if self.array:
            #self.axes[axes_row, 0].set_title(f'mean {data_frame_to_show.metric_long_name.values[0]}: {sensor} - {data_frame_to_show.metric_obs_platform.values[0]}')
            #self.axes[axes_row + 1*iterator, 0].set_title(f'RMS: {data_frame_to_show.metric_long_name.values[0]}: {sensor} - {data_frame_to_show.metric_obs_platform.values[0]}')
            #self.axes[axes_row + 2*iterator, 0].set_title(f'Nobs assimilated: {sensor} - {data_frame_to_show.metric_obs_platform.values[0]}')
    
            for i in range(nmetrics * iterator):
                if i==0:
                    labelbottom=False
                    labeltop=True
                elif i % 8 == 0:
                    labeltop=False
                    labelbottom=True
                else:
                    labeltop=False
                    labelbottom=False
                
                self.axes[axes_row + i, 0].tick_params(axis='x', which='both', top=True, bottom=True,
                                     labeltop=labeltop,labelbottom=labelbottom)
                if self.dark_theme:
                    self.axes[axes_row + i, 0].grid(True)
            
                # Set ticks on both left and right vertical axes
                self.axes[axes_row + i, 0].tick_params(axis='y', which='both', left=True, right=True)
            
                self.axes[axes_row + i, 0].xaxis.set_major_locator(self.locator)
                self.axes[axes_row + i, 0].xaxis.set_major_formatter(self.formatter)
                self.axes[axes_row + i, 0].xaxis.set_minor_locator(self.month_locator)
                
                self.axes[axes_row + i, 0].set_yscale('log')
                self.axes[axes_row +i, 0].invert_yaxis()
                if data_frame_to_show.metric_unit.values[0] == 'percent of qsaturation guess':
                    self.axes[axes_row +i, 0].set_ylim(1100, 350)
                    self.axes[axes_row +i, 0].set_yticks([400, 600, 1000])
                else:
                    self.axes[axes_row +i, 0].set_ylim(1100, 75)
                    self.axes[axes_row +i, 0].set_yticks([100, 200, 400, 600, 1000])
                    
                #self.axes[axes_row +i, 0].set_yticks(np.arange(50, 1100, 50), minor=True)
                self.axes[axes_row+i, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
                self.axes[axes_row + i, 0].set_ylabel('atm p (hPa)')
                
        else:
            self.axes[axes_row, 0].set_title(f'{sensor}, {data_frame_to_show.metric_obs_platform.values[0]}')
            self.axes[axes_row, 1].set_title(f'{sensor}, {data_frame_to_show.metric_obs_platform.values[0]}')
            self.axes[axes_row, 2].set_title(f'{sensor}, {data_frame_to_show.metric_obs_platform.values[0]}')
    

            self.axes[axes_row, 0].tick_params(axis='x', which='both', top=True, bottom=True,
                                     labelbottom=True)
            self.axes[axes_row, 1].tick_params(axis='x', which='both', top=True, bottom=True,
                                     labelbottom=True)
            self.axes[axes_row, 2].tick_params(axis='x', which='both', top=True, bottom=True,
                                     labelbottom=True)
            for col_idx in range(ncols):
                if self.dark_theme:
                    self.axes[axes_row, col_idx].grid(True)
            
                # Set ticks on both left and right vertical axes
                self.axes[axes_row, col_idx].tick_params(axis='y', which='both', left=True, right=True)
            
                self.axes[axes_row, col_idx].xaxis.set_major_locator(self.locator)
                self.axes[axes_row, col_idx].xaxis.set_major_formatter(self.formatter)
                self.axes[axes_row, col_idx].xaxis.set_minor_locator(self.month_locator)
                
            # vertical axes labels
            self.axes[axes_row, 0].set_ylabel(f'Bias {data_frame_to_show.metric_long_name.values[0]}')
            self.axes[axes_row, 1].set_ylabel(f'RMS {data_frame_to_show.metric_long_name.values[0]}')
            self.axes[axes_row, 2].set_ylabel('Number of obs assimilated')

            if self.dark_theme:
                self.axes[axes_row, 0].axhline(color='#A2A4A3', lw=0.5)
            else:
                self.axes[axes_row, 0].axhline(color='black', lw=0.5)
    
    def plot_seasons(self, timeseries_df, plevs_bnds, axes_col, experiment, metric_unit=None,
                     seasonal_mean_diffs=None, xmin=None, xmax=None):
        
        dplevs = np.diff(plevs_bnds)
        plevs = plevs_bnds[:-1] + dplevs/2.
        
        seasons = timeseries_df.resample('QS-DEC').mean()
        
        season_6hr_dict = {
            'DJF' : timeseries_df[timeseries_df.index.month.isin([12,1,2])],
            'MAM' : timeseries_df[timeseries_df.index.month.isin([3,4,5])],
            'JJA' : timeseries_df[timeseries_df.index.month.isin([6,7,8])],
            'SON' : timeseries_df[timeseries_df.index.month.isin([9,10,11])],
        }
        
        season_dict = {
            'DJF' : seasons[seasons.index.month == 12],
            'MAM' : seasons[seasons.index.month == 3],
            'JJA' : seasons[seasons.index.month == 6],
            'SON' : seasons[seasons.index.month == 9],
        }
        season_row_map = {
            'JJA' : 0,
            'MAM' : 1,
            'DJF' : 2,
            'SON' : 3,
        }
        
        if seasonal_mean_diffs is not None:
            seasonal_mean_diffs_dict = {
                'DJF' : seasonal_mean_diffs[seasonal_mean_diffs.index.month == 12],
                'MAM' : seasonal_mean_diffs[seasonal_mean_diffs.index.month == 3],
                'JJA' : seasonal_mean_diffs[seasonal_mean_diffs.index.month == 6],
                'SON' : seasonal_mean_diffs[seasonal_mean_diffs.index.month == 9],
            }
        
        for season, data in season_dict.items():
            mean = data.mean(axis=0)
            std = data.std(axis=0, ddof=1)
            if seasonal_mean_diffs is not None:
                tstat, p_value = get_tstat(seasonal_mean_diffs_dict[season])
                
                if p_value is None:
                    # markup figure if bad values
                    plevs_bnds_sig = plevs_bnds[:-1]
                    dplevs_sig = dplevs
                    sig_edgecolor = '#096FAE'
                    sig_hatch = 'x*'
                    
                else:
                    # markup figure with stippling for significant differences
                    sig = p_value < P_TEST_LEVEL
                    
                    plevs_bnds_sig = plevs_bnds[:-1][sig[:-1]]
                    dplevs_sig = dplevs[sig[:-1]]
                    sig_edgecolor = '#000000'
                    sig_hatch = '...'
                                
                self.axes[season_row_map[season], axes_col].barh(
                    plevs_bnds_sig,
                    align='edge',
                    width=xmax - xmin,
                    height=dplevs_sig,
                    left=xmin,
                    edgecolor=sig_edgecolor,
                    facecolor='none',
                    lw=0.,
                    hatch=sig_hatch,
                    zorder=5,
                )
                            
            self.axes[season_row_map[season], axes_col].fill_betweenx(
                plevs,#plevs_bnds[1:],
                mean[:-1] + 2.*std[:-1],
                x2=mean[:-1] - 2.*std[:-1],
                color=self.config_dict['experiment_plot_dict']
                    [experiment]['color'],
                #step='pre',
                alpha=0.25,
                zorder=2 + self.config_dict['experiment_plot_dict']
                    [experiment]['zorder'],
                lw=0,
                edgecolor='none',
            )
            
            self.axes[season_row_map[season], axes_col].errorbar(
                mean[:-1],
                plevs,
                #edges=plevs_bnds,
                #xerr=std[:-1],
                yerr=-dplevs/2.,
                #orientation='horizontal',
                #baseline=None,
                #ls=self.config_dict['experiment_plot_dict']
                #    [experiment]['ls2'],
                fmt=self.config_dict['experiment_plot_dict']
                    [experiment]['ls2'],
                #marker=self.config_dict['experiment_plot_dict'][experiment]['marker'],
                color=self.config_dict['experiment_plot_dict']
                    [experiment]['color'],
                #mec=self.config_dict['experiment_plot_dict']
                #    [experiment]['color2'],
                #ecolor=self.config_dict['experiment_plot_dict']
                #    [experiment]['color2'],
                lw=1.,
                elinewidth=3.,
                alpha=1,
                label=self.friendly_names_dict[experiment],
                zorder=6 + self.config_dict['experiment_plot_dict']
                    [experiment]['zorder']
            )
                        
            if axes_col == 0:
                self.axes[season_row_map[season], 0].set_ylabel(f'{season}\natm p (hPa)')
                self.axes[season_row_map[season], -1].tick_params(
                    axis='y', which='both', left=True, right=True, labelright=True)
                if season_row_map[season] == 0:
                    self.axes[0, 0].legend(loc='upper left')
                    
            if self.dark_theme:
                self.axes[season_row_map[season], axes_col].grid(True)
                
            self.axes[season_row_map[season], axes_col].tick_params(
                axis='x', which='both', top=True, bottom=True, labelbottom=True)
            self.axes[season_row_map[season], axes_col].tick_params(
                axis='y', which='both', left=True, right=True)
                        
            self.axes[season_row_map[season], axes_col].vlines(
                season_6hr_dict[season].values[:,:-1],
                np.broadcast_to(plevs + dplevs/2., season_6hr_dict[season].values[:,:-1].shape),
                np.broadcast_to(plevs - dplevs/2., season_6hr_dict[season].values[:,:-1].shape),
                colors=self.config_dict['experiment_plot_dict']
                        [experiment]['color2'],
                alpha=0.5,
                zorder=self.config_dict['experiment_plot_dict']
                    [experiment]['zorder']
            )
            
            #self.axes[season_row_map[season], axes_col].set_facecolor('#A2A4A3')
            self.axes[season_row_map[season], axes_col].set_yscale('log')
            self.axes[season_row_map[season], axes_col].invert_yaxis()
            if metric_unit=='%':
                self.axes[season_row_map[season], axes_col].set_ylim(1100,350)
                self.axes[season_row_map[season], axes_col].set_yticks([400, 600, 1000])
            else:
                self.axes[season_row_map[season], axes_col].set_ylim(1100,75)
                self.axes[season_row_map[season], axes_col].set_yticks([100, 200, 400, 600, 1000])
            self.axes[season_row_map[season], axes_col].get_yaxis().set_major_formatter(plt.ScalarFormatter())
    
    def plot_data(self, axes_row, variable, sensor, metric_unit=None,
                     alpha_foreground=0.9,
                     alpha_background=0.1,
                     iterator=None,
                     do_seasons=False):
        
        exp_comp_dict = dict()
        experiment_idx = 0
        if metric_unit == 'percent of qsaturation guess':
            metric_unit = '%'
            bias_cmap = cc.cm.CET_CBD1
        elif metric_unit == 'm/s':
            bias_cmap = cc.cm.CET_D2_r
        else:
            bias_cmap = cc.cm.CET_D1A_r
        
        for experiment, timeseries_dict in self.experiment_timeseries_dict.items():
            exp_comp_dict[experiment_idx] = dict()
            metric_idx = 0
            for metric, timeseries_data in timeseries_dict.items():
                exp_comp_dict[experiment_idx][metric] = dict()
                if sensor in timeseries_data.timestamp_dict[metric].keys():
                    value_arr = timeseries_data.value_dict[metric][sensor]['asm']
                    timestamp_arr = timeseries_data.timestamp_dict[metric][sensor]['asm']
                    if self.array:
                        plevs_bot_arr = np.array([float(p) for p in timeseries_data.pressure_levs_dict[metric][sensor]['asm']['plev_bot']])
                        plevs_top_arr = np.array([float(p) for p in timeseries_data.pressure_levs_dict[metric][sensor]['asm']['plev_top']])
                        plevs_bnds = np.concatenate([plevs_bot_arr[:-1], plevs_top_arr[-2:]])[:-1]
                
                if metric.split('_')[0] == 'bias' and sensor in timeseries_data.timestamp_dict[metric].keys() and len(timestamp_arr) > 0:
                    """ mean error plot
                    """
                    if self.array:
                        pos = 2
                        bias_timeseries = pd.DataFrame(
                            data=np.array(
                                value_arr
                            ),
                            index=timestamp_arr,
                        ).astype(float)
                    
                    else:
                        bias_timeseries = pd.Series(
                            data=np.array(
                                value_arr
                            ),
                            index=timestamp_arr
                        ).astype(float)

                    bias_timeseries = bias_timeseries.combine_first(
                        self.time_domain
                    )
                    
                    if do_seasons:
                        seasonal_mean_diffs = self.diffs_exp_bias.resample('QS-DEC').mean()
                        self.plot_seasons(bias_timeseries, plevs_bnds, pos, experiment, metric_unit=metric_unit,
                                          seasonal_mean_diffs=seasonal_mean_diffs,
                                          xmin=-self.seasonal_bias_xmax,
                                          xmax=self.seasonal_bias_xmax)
                        self.axes[3, pos].set_xlabel(f'bias ({metric_unit})')
                        for season_idx in range(4):
                            if self.dark_theme:
                                self.axes[season_idx, pos].axvline(color='#A2A4A3', lw=0.5)
                            else:
                                self.axes[season_idx, pos].axvline(color='black', lw=0.75, zorder=3)
                                self.axes[season_idx, pos].set_xlim(left=-self.seasonal_bias_xmax, right=self.seasonal_bias_xmax)
                        
                    else:
                        mean_values_smooth = bias_timeseries.rolling(
                            window=self.window_size,
                            min_periods=self.min_periods,
                            center=True,
                        ).mean()
                        
                    if self.array and not do_seasons:
                        exp_comp_dict[experiment_idx][metric][sensor]={'expname':experiment,'values':bias_timeseries}
                        vmax = np.nanquantile(np.abs(mean_values_smooth.values), 0.97725)
                        if vmax > self.seasonal_bias_xmax and experiment_idx == 0:
                            self.seasonal_bias_xmax = vmax
                        pcmesh = self.axes[axes_row + experiment_idx + pos*iterator, 0].pcolormesh(
                            self.time_domain_bnds.index,
                            plevs_bnds,
                            mean_values_smooth.values[:,:-1].T,
                            cmap=bias_cmap,
                            vmax = self.seasonal_bias_xmax,
                            vmin = -self.seasonal_bias_xmax,
                            shading='flat',
                            rasterized=True,
                            )
                        '''
                        self.axes[axes_row + experiment_idx + pos*iterator, 0].set_yscale('log')
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].invert_yaxis()
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].set_yticks([1000, 500, 100])
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].set_yticks(np.arange(50, 1000, 50), minor=True)
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())                        
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].set_ylabel(f'{self.friendly_names_dict[experiment]}\natm. p. (hPa)')
                        '''
                        fig = self.axes[axes_row + experiment_idx+ pos*iterator, 0].get_figure()
                        cbar = fig.colorbar(pcmesh, ax=self.axes[axes_row + experiment_idx+ pos*iterator, 0], aspect=5, pad=0.02)
                        cbar.locator = mticker.MaxNLocator(nbins=5)
                        cbar.update_ticks()
                        cbar.set_label(f'bias ({metric_unit})\n{self.friendly_names_dict[experiment]}')
                        
                        if experiment_idx == 1:
                            diffs = exp_comp_dict[experiment_idx-1][metric][sensor]['values'] - exp_comp_dict[experiment_idx][metric][sensor]['values']
                            self.diffs_exp_bias = diffs
                            values_to_plot = diffs.rolling(
                                window=self.window_size,
                                min_periods=self.min_periods,
                                center=True
                            ).mean()
                            label_to_show = f'{self.friendly_names_dict[exp_comp_dict[experiment_idx-1][metric][sensor]["expname"]]} - {self.friendly_names_dict[exp_comp_dict[experiment_idx][metric][sensor]["expname"]]}'
                            vmax = np.nanquantile(np.abs(values_to_plot.values), 0.97725)
                            pcmesh = self.axes[axes_row+ 1 + experiment_idx+ pos*iterator, 0].pcolormesh(
                                self.time_domain_bnds.index,
                                plevs_bnds,
                                values_to_plot.values[:,:-1].T,
                                cmap=bias_cmap,
                                vmax = vmax,
                                vmin = -vmax,
                                shading='flat',
                                rasterized=True,
                                )
                            
                            '''
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_yscale('log')
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].invert_yaxis()
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_yticks([1000, 500, 100])
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_yticks(np.arange(50, 1000, 50), minor=True)
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_ylabel(f'{label_to_show}\natm. p. (hPa)')
                            '''
                                
                            fig = self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].get_figure()
                            cbar = fig.colorbar(pcmesh, ax=self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0], aspect=5, pad=0.02)
                            cbar.locator = mticker.MaxNLocator(nbins=5)
                            cbar.update_ticks()
                            cbar.set_label(f'bias ({metric_unit})\n{label_to_show}')
                        elif experiment_idx > 1:
                            warnings.warn("inter-experiment differences not supported for more than two experiments")
                    elif not do_seasons:
                    
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

                        self.axes[axes_row, 0].legend(loc='upper right')
                
                elif metric.split('_')[0] == 'rms' and sensor in timeseries_data.timestamp_dict[metric].keys() and len(timestamp_arr) > 0:
                    """RMS error plot
                    """
                    if self.array:
                        pos = 1
                        
                        rmse_timeseries = pd.DataFrame(
                            data=np.array(
                                value_arr
                            ),
                            index=timestamp_arr
                        ).astype(float)
                    else:
                        rmse_timeseries = pd.Series(
                            data=np.array(
                                value_arr
                            ),
                            index=timestamp_arr
                        ).astype(float)
                            
                    rmse_timeseries = rmse_timeseries.combine_first(
                            self.time_domain
                    )
                    
                    if do_seasons:
                        seasonal_mean_diffs = self.diffs_exp_rmsd.resample('QS-DEC').mean()
                        self.plot_seasons(rmse_timeseries, plevs_bnds, axes_row + pos, experiment, metric_unit=metric_unit,
                                          seasonal_mean_diffs=seasonal_mean_diffs,
                                          xmin=0.,
                                          xmax=self.seasonal_rms_xmax)
                        self.axes[3, pos].set_xlabel(f'RMSD ({metric_unit})')
                        for season_idx in range(4):
                            self.axes[season_idx, pos].set_xlim(left=0, right=self.seasonal_rms_xmax)
                            
                        
                    else:
                        rmse_values_smooth = rmse_timeseries.rolling(
                            window=self.window_size,
                            min_periods=self.min_periods,
                            center=True,
                        ).mean()
                    
                    if self.array and not do_seasons:
                        exp_comp_dict[experiment_idx][metric][sensor] = {'expname':experiment,'values':rmse_timeseries}
                        vmax = np.nanquantile(np.abs(rmse_values_smooth.values), 0.97725)
                        if vmax > self.seasonal_rms_xmax and experiment_idx == 0:
                            self.seasonal_rms_xmax = vmax
                        pcmesh = self.axes[axes_row + experiment_idx + pos*iterator, 0].pcolormesh(
                            self.time_domain_bnds.index,
                            plevs_bnds,
                            rmse_values_smooth.values[:,:-1].T,
                            cmap=cc.cm.CET_L3_r,
                            vmax = self.seasonal_rms_xmax,
                            vmin = 0,
                            shading='flat',
                            rasterized=True,
                            )
                        
                        '''
                        self.axes[axes_row + experiment_idx + pos*iterator, 0].set_yscale('log')
                        self.axes[axes_row + experiment_idx + pos*iterator, 0].invert_yaxis()
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].set_yticks([1000, 500, 100])
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].set_yticks(np.arange(50, 1000, 50), minor=True)
                        self.axes[axes_row + experiment_idx + pos*iterator, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
                        self.axes[axes_row + experiment_idx + pos*iterator, 0].set_ylabel(f'{self.friendly_names_dict[experiment]}\natm. p. (hPa)')
                        '''
                        
                        fig = self.axes[axes_row + experiment_idx + pos*iterator, 0].get_figure()
                        cbar = fig.colorbar(pcmesh, ax=self.axes[axes_row + experiment_idx + pos*iterator, 0], aspect=5, pad=0.02)
                        cbar.locator = mticker.MaxNLocator(nbins=5)
                        cbar.update_ticks()
                        cbar.set_label(f'RMSD ({metric_unit})\n{self.friendly_names_dict[experiment]}')
                        
                        if experiment_idx == 1:
                            diffs = exp_comp_dict[experiment_idx-1][metric][sensor]['values'] - exp_comp_dict[experiment_idx][metric][sensor]['values']
                            self.diffs_exp_rmsd = diffs
                            values_to_plot = diffs.rolling(
                                window=self.window_size,
                                min_periods=self.min_periods,
                                center=True,
                            ).mean()
                            label_to_show = f'{self.friendly_names_dict[exp_comp_dict[experiment_idx-1][metric][sensor]["expname"]]} - {self.friendly_names_dict[exp_comp_dict[experiment_idx][metric][sensor]["expname"]]}'
                            vmax = np.nanquantile(np.abs(values_to_plot.values), 0.97725)
                            pcmesh = self.axes[axes_row+ 1 + experiment_idx + pos*iterator, 0].pcolormesh(
                                self.time_domain_bnds.index,
                                plevs_bnds,
                                values_to_plot.values[:,:-1].T,
                                cmap=cc.cm.CET_D7,
                                vmax = vmax,
                                vmin = -vmax,
                                shading='flat',
                                rasterized=True,
                                )
                            
                            '''
                            self.axes[axes_row+1 + experiment_idx + pos*iterator, 0].set_yscale('log')
                            self.axes[axes_row+1 + experiment_idx + pos*iterator, 0].invert_yaxis()
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_yticks([1000, 500, 100])
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_yticks(np.arange(50, 1000, 50), minor=True)
                            self.axes[axes_row+1 + experiment_idx + pos*iterator, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
                            self.axes[axes_row+1 + experiment_idx + pos*iterator, 0].set_ylabel(f'{label_to_show}\natm. p. (hPa)')
                            '''
                        
                            fig = self.axes[axes_row+1 + experiment_idx + pos*iterator, 0].get_figure()
                            cbar = fig.colorbar(pcmesh, ax=self.axes[axes_row+1 + experiment_idx + pos*iterator, 0], aspect=5, pad=0.02)
                            cbar.locator = mticker.MaxNLocator(nbins=5)
                            cbar.update_ticks()
                            cbar.set_label(f'RMSD ({metric_unit})\n{label_to_show}')
                        elif experiment_idx > 1:
                            warnings.warn("inter-experiment differences not supported for more than two experiments")
                        
                    elif not do_seasons:
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
                    if self.array:
                        pos = 0
                        
                        nobs_used_timeseries = pd.DataFrame(
                            data=np.array(
                                value_arr
                            ),
                            index=timestamp_arr
                        ).astype(float)
                    else:
                        nobs_used_timeseries = pd.Series(
                            data=np.array(
                                value_arr
                            ),
                            index=timestamp_arr
                        ).astype(float)

                    nobs_used_timeseries = nobs_used_timeseries.combine_first(
                        self.time_domain
                    )
                    
                    if do_seasons:
                        self.plot_seasons(nobs_used_timeseries, plevs_bnds, axes_row + pos, experiment,metric_unit=metric_unit)
                        self.axes[3, pos].set_xlabel('n obs')
                        for season_idx in range(4):
                            self.axes[season_idx, pos].set_xlim(left=0, right=self.seasonal_nobs_xmax)
                        
                    else:
                        nobs_used_smooth = nobs_used_timeseries.rolling(
                            window=self.window_size,
                            min_periods=self.min_periods,
                            center=True,
                        ).mean()
                        
                    if self.array and not do_seasons:
                        exp_comp_dict[experiment_idx][metric][sensor] = {'expname':experiment,'values':nobs_used_timeseries}
                        vmax = np.nanquantile(np.abs(nobs_used_smooth.values), 0.9)#0.84)
                        if vmax > self.seasonal_nobs_xmax and experiment_idx == 0:
                            self.seasonal_nobs_xmax = vmax
                        pcmesh = self.axes[axes_row + experiment_idx + pos*iterator, 0].pcolormesh(
                            self.time_domain_bnds.index,
                            plevs_bnds,
                            nobs_used_smooth.values[:,:-1].T,
                            cmap=cc.cm.CET_L1_r,
                            vmax = self.seasonal_nobs_xmax,
                            vmin = 0,
                            shading='flat',
                            rasterized=True,
                            )
                        '''
                        self.axes[axes_row + experiment_idx +pos*iterator, 0].set_yscale('log')
                        self.axes[axes_row + experiment_idx +pos*iterator, 0].invert_yaxis()
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].set_yticks([1000, 500, 100])
                        self.axes[axes_row + experiment_idx+ pos*iterator, 0].set_yticks(np.arange(50, 1000, 50), minor=True)
                        self.axes[axes_row + experiment_idx +pos*iterator, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
                        self.axes[axes_row + experiment_idx +pos*iterator, 0].set_ylabel(f'{self.friendly_names_dict[experiment]}\natm. p. (hPa)')
                        '''
                        
                        fig = self.axes[axes_row + experiment_idx +pos*iterator, 0].get_figure()
                        cbar = fig.colorbar(pcmesh, ax=self.axes[axes_row + experiment_idx +pos*iterator, 0], aspect=5, pad=0.02)
                        cbar.locator = mticker.MaxNLocator(nbins=5)
                        cbar.update_ticks()
                        cbar.set_label(f'n obs\n{self.friendly_names_dict[experiment]}')
                        
                        if experiment_idx == 1:
                            diffs = exp_comp_dict[experiment_idx-1][metric][sensor]['values'] - exp_comp_dict[experiment_idx][metric][sensor]['values']
                            values_to_plot = diffs.rolling(
                                window=self.window_size,
                                min_periods=self.min_periods,
                                center=True,
                            ).mean()
                            label_to_show = f'{self.friendly_names_dict[exp_comp_dict[experiment_idx-1][metric][sensor]["expname"]]} - {self.friendly_names_dict[exp_comp_dict[experiment_idx][metric][sensor]["expname"]]}'
                            vmax = np.nanquantile(np.abs(values_to_plot.values), 0.9)#0.84)
                            pcmesh = self.axes[axes_row+ 1 + experiment_idx +pos*iterator, 0].pcolormesh(
                                self.time_domain_bnds.index,
                                plevs_bnds,
                                values_to_plot.values[:,:-1].T,
                                cmap=cc.cm.CET_D7_r,
                                vmax = vmax,
                                vmin = -vmax,
                                shading='flat',
                                rasterized=True,
                                )
                            
                            '''
                            self.axes[axes_row+1 + experiment_idx +pos*iterator, 0].set_yscale('log')
                            self.axes[axes_row+1 + experiment_idx +pos*iterator, 0].invert_yaxis()
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_yticks([1000, 500, 100])
                            self.axes[axes_row+1 + experiment_idx+ pos*iterator, 0].set_yticks(np.arange(50, 1000, 50), minor=True)
                            self.axes[axes_row+1 + experiment_idx +pos*iterator, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
                            self.axes[axes_row+1 + experiment_idx + pos*iterator, 0].set_ylabel(f'{label_to_show}\natm. p. (hPa)')
                            '''
                            
                            fig = self.axes[axes_row+1 + experiment_idx +pos*iterator, 0].get_figure()
                            cbar = fig.colorbar(pcmesh, ax=self.axes[axes_row+1 + experiment_idx +pos*iterator, 0], aspect=5, pad=0.02)
                            cbar.locator = mticker.MaxNLocator(nbins=5)
                            cbar.update_ticks()
                            cbar.set_label(f'n obs\n{label_to_show}')
                        elif experiment_idx > 1:
                            warnings.warn("inter-experiment differences not supported for more than two experiments")
                        
                    elif not do_seasons:
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
                            zorder=3)
            
            experiment_idx += 1                

def prun(experiment_list=None, sensor_list=None, variable_list=None, start_date=None, stop_date=None,
         pressure_bins=False):    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    args = parse_arguments()
    gsi_stage = args.gsi_stage

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
    
    if args.pressure_bins:
        pressure_bins=True
    if pressure_bins:
        metric_name_key = 'metric_name'
    else:
        metric_name_key = 'name'
            
    # Rank 0 prepares the data
    if rank == 0:
        global_data_frame = get_data_frame(
            experiment_list,
            sensor_list,
            variable_list,
            start_date=start_date,
            stop_date=stop_date,
            gsi_it=gsi_stage,
            pressure_bins=pressure_bins)

        # Split the data by variable (one part per variable)
        data_frame_parts_dict = dict()
        
        if pressure_bins:
            for sensor in sensor_list:
                data_frame_parts_dict[sensor] = global_data_frame[
                    global_data_frame[metric_name_key].str.contains(str(sensor))]

    else:
        data_frame_parts_dict = None

    if pressure_bins:
        # Calculate how many variables each process should handle
        sensors_per_process = len(sensor_list) // size

        # Handle leftover sensors (remaining variables are distributed to the first few processes)
        leftover_sensors = len(sensor_list) % size
    
        if rank == 0:
            for i in range(0, size):
                # Calculate the subset of variables for this rank
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
                # Only work on data for the specific variable assigned to the process
                '''
                print(f"Rank {rank} is processing variable: {var} and here is "
                       f"the data frame: {data_frame.metric_type}")
            
                '''
                experiment_metrics_timeseries_data = GSIConvFit2ObsFig(
                    data_frame=data_frame,
                    input_data_frame=True,
                    gsi_it=gsi_stage,
                    pressure_bins=pressure_bins
                )
            
                experiment_metrics_timeseries_data.variable_list = variable_list
                experiment_metrics_timeseries_data.config_dict['sensor_list'] = sensor_list
                experiment_metrics_timeseries_data.sensor_id = sensor
                experiment_metrics_timeseries_data.experiment_list = experiment_list
                experiment_metrics_timeseries_data.config_dict['start_date'] = start_date
                experiment_metrics_timeseries_data.config_dict['stop_date'] = stop_date
                experiment_metrics_timeseries_data.build_timeseries(interactive_figure=args.interactive,
                                                                    days_to_smooth=args.days_to_smooth,
                                                                    da_cycle=args.da_cycle,
                                                                    dark_theme=args.dark_theme)
            
    elif rank==0: # no parallelization
        if global_data_frame is not None:
        
            experiment_metrics_timeseries_data = GSIConvFit2ObsFig(
                data_frame=global_data_frame,
                input_data_frame=True,
                gsi_it=gsi_stage,
                pressure_bins=pressure_bins
            )
        
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
