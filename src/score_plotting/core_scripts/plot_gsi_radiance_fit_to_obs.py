#!/usr/bin/env python

"""
"""

import os
import pathlib
import warnings
import argparse

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from mpi4py import MPI

import gsistats_timeseries
from instrument_channel_nums import get_instrument_channels
import satellite_names

HOURS_PER_DAY = 24. # hours
DA_CYCLE = 6. # hours
DAYS_TO_SMOOTH = 8. # days

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='script to create GSI analysis timeseries figures for '
                    'radiance error monitoring')
    parser.add_argument('figure_output_path', type=str,
                        help='path to where figures will be saved.')
    args = parser.parse_args()

    return args

def config():
    args = parse_arguments()    
    config_dict = {
        'config_path':
            os.path.join(pathlib.Path(__file__).parent.parent.resolve(),
                         'style_lib'),
        'config_file': ['full_3x3pg.mplstyle'],
        'output_path': args.figure_output_path,
        'experiment_list': ['GDAS',
                            'replay_observer_diagnostic_v1',
                            'NASA_GEOSIT_GSISTATS',
                            'scout_run_v1'
                         ],
        'color_list': ['#CFB87C', '#0085CA', '#E4002B', 'black'],
        #'ls_list': [':', '-.', '--', '-'],
        'ls_list': ['-', '-.', '-', '-'],
        'lw_list': [2.5, 2.0, 1.5, 1.0],
        'sensor_list': get_instrument_channels().keys(),#['amsua'],
        'start_date': '1979-01-01 00:00:00',
        'stop_date': '2026-01-01 00:00:00',
    }
    
    '''
    could this be done by string matching for the std/bias etc part? we could
    have a basic friendly dict for that
    '''
    friendly_names_dict={"scout_run_v1": "scout run (3DVar)",
                         "NASA_GEOSIT_GSISTATS": "GEOS-IT",
                         "GDAS": "GDAS",
                         "replay_observer_diagnostic_v1": "UFS Replay",
                         "std_GSIstage_1": "STD",
                         "variance_GSIstage_1": "obs error variance",
                         "bias_post_corr_GSIstage_1": "ME",
                         "sqrt_bias_GSIstage_1": "RMSE"}
                         
    return(config_dict, friendly_names_dict)

def get_data_frame(experiment_list, sensor_list,
                   start_date='1979-01-01 00:00:00',
                   stop_date='2026-01-01 00:00:00'):
        
    array_metric_list = list()
    for sensor in sensor_list:
        array_metric_list.append(f'{sensor}_bias_post_corr_GSIstage_1')
        array_metric_list.append(f'{sensor}_std_GSIstage_1')
        array_metric_list.append(f'{sensor}_variance_GSIstage_1')
        array_metric_list.append(f'{sensor}_sqrt_bias_GSIstage_1')
        array_metric_list.append(f'{sensor}_nobs_used_GSIstage_1')
        array_metric_list.append(f'{sensor}_nobs_tossed_GSIstage_1')
        array_metric_list.append(f'{sensor}_use_GSIstage_None')
        
    return gsistats_timeseries.get_data_frame(
                               experiment_list,
                               array_metric_list,
                               start_date=start_date,
                               stop_date=stop_date,
                               select_sat_name=False,
                               sat_name=None)

class GSIRadianceFit2ObsFig(object):
    """
    """
    def __init__(self, data_frame=None, input_data_frame=False):
        """
        """
        self.config_dict, self.friendly_names_dict = config()
        self.channel_dict = get_instrument_channels()
        self.experiment_list = self.config_dict['experiment_list']
        
        if self.config_dict['config_path'] and self.config_dict['config_file']:
            for style_file in self.config_dict['config_file']:
                style_file_path = os.path.join(self.config_dict['config_path'],
                                               style_file)
                plt.style.use(style_file_path)
                
        if input_data_frame:
            self.data_frame = data_frame
        else:
            self.data_frame = get_data_frame(self.experiment_list,
                                             self.config_dict['start_date'],
                                             self.config_dict['stop_date'])
    
    def build_timeseries(self):
        for sensor, channel_list in self.channel_dict.items():
            if sensor in self.config_dict['sensor_list']:
                experiment_timeseries_datetime_init=None
                array_metric_list = [f'{sensor}_bias_post_corr_GSIstage_1',
                                     f'{sensor}_std_GSIstage_1',
                                     f'{sensor}_variance_GSIstage_1',
                                     f'{sensor}_sqrt_bias_GSIstage_1',
                                     f'{sensor}_nobs_used_GSIstage_1',
                                     f'{sensor}_nobs_tossed_GSIstage_1',
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
                    init_datetime=experiment_timeseries_datetime_init)
                        
    def make_figures(self, sensor, ncols=3, init_datetime=None,
                     alpha_foreground=0.9,
                     alpha_background=0.25):
        output_dir = os.path.join(self.config_dict['output_path'], f"{sensor}")
        window_size = int(DAYS_TO_SMOOTH * (HOURS_PER_DAY / DA_CYCLE))
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
        
        for channel_idx, channel_num in enumerate(self.channel_dict[sensor]):
            if len(sat_set) > 0:
                max_yerr=0.1 # temperature (K)
                fig, axes = plt.subplots(len(sat_set), ncols, sharex=True,sharey=False,
                                         squeeze=False,
                                         figsize=(2*ncols*3.74, len(sat_set)*4.53))
                title_str0 = f"GSI radiance data analysis fit to observations (O-B) [metrics downloaded from RDB {self.db_name}"
                
                if init_datetime:
                    init_ctime = init_datetime.ctime()
                    title_str1 = f" {init_ctime}]"
                
                else:
                    title_str1 = "]"
                    
                fig.suptitle(f"{title_str0}{title_str1}")
                #axes[-1, 0].set_xlabel = 'cycle date (Gregorian)'
                #axes[-1, 1].set_xlabel = 'cycle date (Gregorian)'
            
                for row, sat_sensor in enumerate(sorted(sat_set)):
                    sat_short_name = sat_sensor.split('_')[:-1][0]
                    sat_label = satellite_names.get_longname(sat_short_name)
                    
                    # subplot titles
                    axes[row, 0].set_title(f"{sat_label} {sensor} channel {channel_num}")
                    axes[row, 1].set_title(f"{sat_label} {sensor} channel {channel_num}")
                    axes[row, 2].set_title(f"{sat_label} {sensor} channel {channel_num}")
                    
                    # vertical axes labels
                    axes[row, 0].set_ylabel('Temperature mean error (K)')
                    axes[row, 1].set_ylabel('Temperature RMS error (K)')
                    axes[row, 2].set_ylabel('Number of observations tossed')
                    rejection_ratio_ax = axes[row, 2].twinx()
                    rejection_ratio_ax.set_ylabel('Percentage of observations tossed (%)')
                    
                    axes[row, 0].axhline(color='black', lw=0.5)

                    rejection_ratio_ax.set_ylim(0, 100)
                    rejection_ratio_ax.set_yticks(np.arange(0, 100.1, 20))                    
                    rejection_ratio_ax.set_yticks(np.arange(0, 100.1, 5), minor=True)

                    # Set ticks on both left and right vertical axes
                    axes[row, 0].tick_params(axis='y', which='both', left=True, right=True)
                    axes[row, 1].tick_params(axis='y', which='both', left=True, right=True)
                    
                    axes[row, 0].tick_params(axis='x', which='both', top=True, bottom=True)
                    axes[row, 1].tick_params(axis='x', which='both', top=True, bottom=True)
                
                    experiment_idx = 0
                    for experiment, timeseries_dict in self.experiment_timeseries_dict.items():
                        for full_stat_name, timeseries_data in timeseries_dict.items():
                            for stat_label, value_dict in timeseries_data.value_dict.items():
                                if stat_label == 'bias_post_corr_GSIstage_1' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    """ mean error plot
                                    """
                                    bias_timestamps = timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    bias_values = np.array(value_dict[sat_sensor])[:,channel_idx]
                                    std_timestamps = timeseries_dict[
                                        f'{sensor}_std_GSIstage_1'].timestamp_dict[
                                            'std_GSIstage_1'][sat_sensor]
                                    std_values = timeseries_dict[
                                        f'{sensor}_std_GSIstage_1'].value_dict[
                                            'std_GSIstage_1'][sat_sensor]
                                            
                                    nobs_used_timestamps = timeseries_dict[
                                        f'{sensor}_nobs_used_GSIstage_1'
                                        ].timestamp_dict['nobs_used_GSIstage_1'
                                            ][sat_sensor]
                                    nobs_used_values = timeseries_dict[
                                        f'{sensor}_nobs_used_GSIstage_1'
                                        ].value_dict[f'nobs_used_GSIstage_1'
                                            ][sat_sensor]
                                    use_timestamps = timeseries_dict[
                                        f'{sensor}_use_GSIstage_None'].timestamp_dict[
                                            'use_GSIstage_None'][sat_sensor]
                                    use_values = timeseries_dict[
                                        f'{sensor}_use_GSIstage_None'].value_dict[
                                            'use_GSIstage_None'][sat_sensor]
                                        
                                    yerrs=list()
                                    nobs_used_arr=list()
                                    use_flags=list()
                                    for time_idx, bias_timestamp in enumerate(bias_timestamps):
                                        if bias_timestamp in std_timestamps:
                                            std_time_idx = std_timestamps.index(bias_timestamp)
                                            yerr = np.array(std_values)[std_time_idx, channel_idx]
                                            
                                            if yerr:
                                                yerrs.append(yerr)
                                            else:
                                                yerrs.append(np.nan)
                                        else:
                                            yerrs.append(np.nan)
                                        
                                        if bias_timestamp in nobs_used_timestamps:
                                            nobs_used_time_idx = nobs_used_timestamps.index(bias_timestamp)
                                            nobs_used_channel = np.array(nobs_used_values)[nobs_used_time_idx, channel_idx]
                                            
                                            if nobs_used_channel:
                                                nobs_used_arr.append(nobs_used_channel)
                                            else:
                                                nobs_used_arr.append(np.nan)
                                        else:
                                            nobs_used_arr.append(np.nan)
                                        
                                        if bias_timestamp in use_timestamps:
                                            use_time_idx = use_timestamps.index(bias_timestamp)
                                            use_flag = np.array(use_values)[use_time_idx, channel_idx]
                                            
                                            if use_flag:
                                                use_flags.append(use_flag)
                                            else:
                                                use_flags.append(np.nan)
                                        else:
                                            use_flags.append(np.nan)
                                            
                                    use_flags_plot = np.array([np.nan if x is None else float(x) for x in use_flags])
                                    mean_values_plot = np.ma.masked_where(
                                        use_flags_plot < 1,
                                        np.array([np.nan if x is None else float(x) for x in bias_values])
                                    )
                                    
                                    yerrs_plot = np.ma.masked_where(
                                        use_flags_plot < 1,
                                        np.array([np.nan if x is None else float(x) for x in yerrs])
                                    )
                                    mean_values_smooth = pd.Series(
                                        mean_values_plot,
                                        index=bias_timestamps)
                                    #nobs_used_plot = np.array([np.nan if x is None else float(x) for x in nobs_used_arr])
                                    #standard_errs = np.array(yerrs_plot) / np.sqrt(nobs_used_plot)
                                        
                                    axes[row, 0].bar(
                                        bias_timestamps,
                                        2.*yerrs_plot,
                                        width=pd.Timedelta(hours=DA_CYCLE),
                                        bottom=mean_values_plot - yerrs_plot,
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=alpha_background
                                    )
                                
                                    '''
                                    axes[row, 0].barh(np.clip(mean_values_plot,
                                                              YMIN,
                                                              YMAX),
                                                      pd.Timedelta(hours=2),
                                                      height=0.1,
                                                      left=bias_timestamps - pd.Timedelta(hours=1),
                                                      color=self.config_dict['color_list'][experiment_idx],
                                                      alpha = 1.0)
                                    '''
                                    axes[row, 0].plot(
                                        bias_timestamps,
                                        mean_values_plot,
                                        marker='none',
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=alpha_background,
                                        lw=0.5,
                                        ls='-',
                                    )
                                    axes[row, 0].plot(
                                        bias_timestamps,
                                        mean_values_smooth.rolling(
                                            window=window_size,
                                            center=True,
                                            win_type='triang').mean(),
                                        marker='none',
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=alpha_foreground,
                                        lw=self.config_dict['lw_list'][experiment_idx],
                                        ls=self.config_dict['ls_list'][experiment_idx],
                                        label=self.friendly_names_dict[experiment]
                                    )
                                    '''
                                    axes[row, 0].errorbar(
                                        bias_timestamps,
                                        np.clip(mean_values_plot, YMIN, YMAX),
                                        xerr=pd.Timedelta(hours=3),
                         fmt='none',#self.config_dict['ls_list'][experiment_idx],
                        #lw=self.config_dict['lw_list'][experiment_idx],
                        elinewidth=self.config_dict['lw_list'][experiment_idx],
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha = 1.0,
                                    )
                                    '''
                                    axes[row,0].legend(loc='upper left')
                            
                                elif stat_label == 'sqrt_bias_GSIstage_1' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    """RMS error plot
                                    """
                                    rmse_timestamps = timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    rmse_values = np.array(value_dict[sat_sensor])[:,channel_idx]
                                    obs_err_var_timestamps = timeseries_dict[
                                        f'{sensor}_variance_GSIstage_1'].timestamp_dict[
                                            'variance_GSIstage_1'][sat_sensor]
                                    obs_err_var_values = timeseries_dict[
                                        f'{sensor}_variance_GSIstage_1'].value_dict[
                                            'variance_GSIstage_1'][sat_sensor]
                                    use_timestamps = timeseries_dict[
                                        f'{sensor}_use_GSIstage_None'].timestamp_dict[
                                            'use_GSIstage_None'][sat_sensor]
                                    use_values = timeseries_dict[
                                        f'{sensor}_use_GSIstage_None'].value_dict[
                                            'use_GSIstage_None'][sat_sensor]
                                        
                                    yerrs2=list()
                                    use_flags=list()
                                    for time_idx, rmse_timestamp in enumerate(rmse_timestamps):
                                        if rmse_timestamp in obs_err_var_timestamps:
                                            obs_err_var_time_idx = obs_err_var_timestamps.index(rmse_timestamp)
                                            yerr2 = np.array(obs_err_var_values)[obs_err_var_time_idx, channel_idx]
                                            
                                            if yerr2:
                                                yerrs2.append(yerr)
                                            else:
                                                yerrs2.append(0)
                                        else:
                                            yerrs2.append(0)
                                            
                                        if rmse_timestamp in use_timestamps:
                                            use_time_idx = use_timestamps.index(rmse_timestamp)
                                            use_flag = np.array(use_values)[use_time_idx, channel_idx]
                                            
                                            if use_flag:
                                                use_flags.append(use_flag)
                                            else:
                                                use_flags.append(np.nan)
                                        else:
                                            use_flags.append(np.nan)
                                
                                    yerrs_plot = np.sqrt(np.array([np.nan if x is None else float(x) for x in yerrs2]))
                                    use_flags_plot = np.array([np.nan if x is None else float(x) for x in use_flags])
                                    max_yerr = np.max(yerrs_plot, initial=max_yerr)
                                    rmse_values_plot = np.ma.masked_where(
                                        use_flags_plot < 1,
                                        np.array([np.nan if x is None else float(x) for x in rmse_values])
                                    )
                                    rmse_values_smooth = pd.Series(rmse_values_plot,
                                                            index=rmse_timestamps)
                                
                                    '''
                                    axes[row, 1].bar(
                                        rmse_timestamps,
                                        np.ma.masked_where(use_flags_plot < 1, 2.*yerrs_plot),
                                        width=pd.Timedelta(hours=DA_CYCLE),
                                        bottom=rmse_values_plot - yerrs_plot,
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=alpha_background
                                    )
                                    '''
                                    
                                    axes[row, 1].plot(
                                        rmse_timestamps,
                                        rmse_values_plot,
                                        marker='none',
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=alpha_background,
                                        lw=0.5,
                                        ls='-',
                                        #xerr=pd.Timedelta(hours=3),
                 #fmt='none',#,self.config_dict['ls_list'][experiment_idx],
                    #lw=self.config_dict['lw_list'][experiment_idx],
                    #elinewidth=self.config_dict['lw_list'][experiment_idx],
                                    )
                                    axes[row, 1].plot(
                                        rmse_timestamps,
                                        rmse_values_smooth.rolling(
                                            window=window_size,
                                            center=True,
                                            win_type='triang'
                                        ).mean(),
                                        marker='none',
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=alpha_foreground,
                                        lw=self.config_dict['lw_list'][experiment_idx],
                                        ls=self.config_dict['ls_list'][experiment_idx],
                                        label=self.friendly_names_dict[experiment],
                                    )
                                    axes[row,1].legend(loc='upper left')
                            
                                elif stat_label == 'nobs_tossed_GSIstage_1' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    """ nobs tossed and rejection ratio plot
                                    """
                                    nobs_tossed_timestamps = timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    nobs_tossed_values = np.array(value_dict[sat_sensor])[:,channel_idx]
                                    nobs_used_timestamps = timeseries_dict[
                                        f'{sensor}_nobs_used_GSIstage_1'
                                        ].timestamp_dict['nobs_used_GSIstage_1'
                                            ][sat_sensor]
                                    nobs_used_values = timeseries_dict[
                                        f'{sensor}_nobs_used_GSIstage_1'
                                        ].value_dict[f'nobs_used_GSIstage_1'
                                            ][sat_sensor]
                                        
                                    nobs_used_arr=list()
                                    nobs_tossed_arr = list()
                                    for time_idx, nobs_tossed_timestamp in enumerate(nobs_tossed_timestamps):
                                        if nobs_tossed_timestamp in nobs_used_timestamps:
                                            nobs_used_time_idx = nobs_used_timestamps.index(nobs_tossed_timestamp)
                                            nobs_used_channel = np.array(nobs_used_values)[nobs_used_time_idx, channel_idx]
                                            if nobs_used_channel:
                                                nobs_used_arr.append(nobs_used_channel)
                                                
                                            else:
                                                nobs_used_arr.append(np.nan)
                                        else:
                                            nobs_used_arr.append(np.nan)
                                            
                                    nobs_tossed_plot = np.array([np.nan if x is None else float(x) for x in nobs_tossed_values])
                                    nobs_used_plot = np.array([np.nan if x is None else float(x) for x in nobs_used_arr])
                                    
                                    rejection_rate = nobs_tossed_plot / (
                                        nobs_used_plot + nobs_tossed_plot)
                                        
                                    axes[row, 2].bar(
                                                nobs_tossed_timestamps,
                                                nobs_tossed_plot,
                                                width=pd.Timedelta(hours=DA_CYCLE),
                                                color=self.config_dict['color_list'][experiment_idx],
                                                alpha=alpha_background,
                                                label=f"n tossed ({self.friendly_names_dict[experiment]})"
                                            )
                                
                                    rejection_ratio_ax.plot(
                                                nobs_tossed_timestamps,
                                                100.*rejection_rate,
                                                marker='none',
                                                color=self.config_dict['color_list'][experiment_idx],
                                                alpha=alpha_foreground,
                                                lw=self.config_dict['lw_list'][experiment_idx],
                                                ls=self.config_dict['ls_list'][experiment_idx],
                                                label=f"{self.friendly_names_dict[experiment]}"
                                            )

                                    #axes[row,2].legend(loc='upper left')
                                    rejection_ratio_ax.legend(loc='upper right')

                                axes[row, 0].xaxis.set_major_formatter(
                                    mdates.ConciseDateFormatter(
                                              axes[row, 0].xaxis.get_major_locator()))
                                axes[row, 1].xaxis.set_major_formatter(
                                    mdates.ConciseDateFormatter(
                                              axes[row, 1].xaxis.get_major_locator()))
                                axes[row, 2].xaxis.set_major_formatter(
                                    mdates.ConciseDateFormatter(
                                              axes[row, 2].xaxis.get_major_locator()))
                                          
                        experiment_idx += 1
                
                for row, sat_sensor in enumerate(sorted(sat_set)):
                    # set ylim, ticks
                    axes[row, 1].set_yticks(np.arange(0, 3.*max_yerr + 0.1, 0.1), minor=True)
                    axes[row, 0].set_yticks(np.arange(-1.5*max_yerr, 1.5*max_yerr + 0.1, 0.1), minor=True)
                
                    axes[row, 0].set_yticks(np.arange(np.around(-1.5*max_yerr - 1), np.around(1.5*max_yerr + 2), 0.5))
                    axes[row, 1].set_yticks(np.arange(0, np.around(3.*max_yerr + 2), 0.5))
                    
                    axes[row, 0].set_ylim(-1.5*max_yerr, 1.5*max_yerr)
                    axes[row, 1].set_ylim(0, 3.*max_yerr)
                
                plt.savefig(os.path.join(output_dir, 
                                  f'gsi_radiance_omb_{sensor}_ch{channel_num}.png'),
                            dpi=300)
                plt.close()

def run_microwave_sounders(sensor_list=['amsua', 'amsub', 'atms', 'ssmis']):
    prun(sensor_list=sensor_list)

def run_atms(sensor_list=['atms']):
    prun(sensor_list=sensor_list)

def run_airs(sensor_list=['airs']):
    prun(sensor_list=sensor_list)

def run_tovs(sensor_list = ['hirs2', 'hirs3', 'hirs4', 'ssu', 'msu']):
    prun(sensor_list = sensor_list)

def run_avhrr(sensor_list = ['avhrr2', 'avhrr3']):
    prun(sensor_list=sensor_list)

def prun(sensor_list=None):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load global configurations and friendly names
    global_config_dict, global_friendly_names_dict = config()

    if sensor_list==None:
        sensor_list = list()
        for sensor in global_config_dict['sensor_list']:
            sensor_list.append(sensor)

    # Rank 0 prepares the data
    if rank == 0:
        global_data_frame = get_data_frame(
            global_config_dict['experiment_list'],
            sensor_list,
            start_date=global_config_dict['start_date'],
            stop_date=global_config_dict['stop_date']
        )

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
                input_data_frame=True
            )
            experiment_metrics_timeseries_data.config_dict['sensor_list'] = [sensor]
            experiment_metrics_timeseries_data.build_timeseries()

def prun0(sensor_list=None):
    # Initialize MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes

    # Load global configurations and friendly names
    global_config_dict, global_friendly_names_dict = config()
    
    if sensor_list==None:
        sensor_list = list()
        for sensor in global_config_dict['sensor_list']:
            sensor_list.append(sensor)
    
    if rank == 0:
        global_data_frame = get_data_frame(global_config_dict['experiment_list'],
                                           sensor_list,
                                           start_date=global_config_dict['start_date'],
                                           stop_date=global_config_dict['stop_date'])
    else:
        global_data_frame = None
    
    global_data_frame = comm.bcast(global_data_frame, root=0)
    
    # Calculate how many sensors each process should handle
    sensors_per_process = len(sensor_list) // size

    # Handle leftover sensors (remaining sensors are distributed to the first few processes)
    leftover_sensors = len(sensor_list) % size

    # Calculate the start and end indices for each process
    start_idx = rank * sensors_per_process + min(rank, leftover_sensors)  # Adjust start index for extra sensors
    end_idx = start_idx + sensors_per_process + (1 if rank < leftover_sensors else 0)  # Adjust end index for extra sensors

    # Slice the sensor list for this process
    local_sensor_list = sensor_list[start_idx:end_idx]
    
    # Each process handles its portion of the sensor list
    for sensor in local_sensor_list:
        experiment_metrics_timeseries_data = GSIRadianceFit2ObsFig(
            data_frame=global_data_frame[
                global_data_frame['metric_instrument_name']==sensor],
            input_data_frame=True
        )
        
        # Set the current sensor for the experiment
        experiment_metrics_timeseries_data.config_dict['sensor_list'] = [sensor]
        #print(f'{sensor}', rank)
        
        # Build time series data for the current sensor
        experiment_metrics_timeseries_data.build_timeseries()

def main():
    """
    """
    #run_avhrr()
    #run_tovs()
    #run_microwave_sounders()
    #run_atms()
    prun()

if __name__ == "__main__":
    main()
