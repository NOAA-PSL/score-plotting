#!/usr/bin/env python

"""
"""

import os
import pathlib
import warnings

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from mpi4py import MPI

from gsistats_timeseries import GSIStatsTimeSeries
from instrument_channel_nums import get_instrument_channels
import satellite_names

def config():
    config_dict = {
        'config_path':
            os.path.join(pathlib.Path(__file__).parent.parent.resolve(),
                         'style_lib'),
        'config_file': 'agu_full_3pg.mplstyle',
        'output_path':
            os.path.join('/', 'media', 'darr', 'results', 'figures',
                         'brightness_temperature_error_timeseries'),
        'experiment_list': ['NASA_GEOSIT_GSISTATS',
                            'GDAS',
                            'replay_observer_diagnostic_v1',
                            'scout_run_v1'
                         ],
        'color_list': ['#E4002B', '#A2A4A3', '#003087', '#0085CA'],
        #'fmt_list': ['|', "1", "2"],
        'ls_list': [':', '-.', '--', '-'],
        'lw_list': [0.75, 1.0, 1.25, 1.5],
        'sensor_list': get_instrument_channels().keys(),#['amsua'],
        'start_date': '1979-01-01 00:00:00',
        'stop_date': '2025-01-01 00:00:00',
    }
    
    '''
    could this be done by string matching for the std/bias etc part? we could
    have a basic friendly dict for that
    '''
    friendly_names_dict={"scout_run_v1": "NOAA atmo-scout",
                         "NASA_GEOSIT_GSISTATS": "NASA GEOS-IT",
                         "GDAS": "GDAS",
                         "replay_observer_diagnostic_v1": "NOAA ROD",
                         "std_GSIstage_1": "STD",
                         "variance_GSIstage_1": "obs error variance",
                         "bias_post_corr_GSIstage_1": "ME",
                         "sqrt_bias_GSIstage_1": "RMSE"}
                         
    return(config_dict, friendly_names_dict)

class GSIRadianceFit2ObsFig(object):
    """
    """
    def __init__(self):
        self.config_dict, self.friendly_names_dict = config()
        self.channel_dict = get_instrument_channels()
        self.experiment_list = self.config_dict['experiment_list']
        
        if self.config_dict['config_path'] and self.config_dict['config_file']:
            style_file = os.path.join(self.config_dict['config_path'],
                                      self.config_dict['config_file'])
            plt.style.use(style_file)
    
    def build_timeseries(self):
        for sensor, channel_list in self.channel_dict.items():
            if sensor in self.config_dict['sensor_list']:
                array_metric_list = [f'{sensor}_bias_post_corr_GSIstage_1',
                                     f'{sensor}_std_GSIstage_1',
                                     f'{sensor}_variance_GSIstage_1',
                                     f'{sensor}_sqrt_bias_GSIstage_1',
                                     #f'{sensor}_nobs_used_GSIstage_1',
                                     #f'{sensor}_nobs_tossed_GSIstage_1'
                                 ]
                                 
                self.experiment_timeseries_dict = dict()
                for experiment in self.config_dict['experiment_list']:
                    self.experiment_timeseries_dict[experiment] = dict()
                    for array_metric in array_metric_list:
                        try:
                            self.experiment_timeseries_dict[
                                experiment][array_metric] = GSIStatsTimeSeries(
                                                self.config_dict['start_date'],
                                                self.config_dict['stop_date'],
                                                experiment_name=experiment,
                                                select_array_metric_types=True,
                                                array_metric_types=array_metric)
                            self.experiment_timeseries_dict[experiment][array_metric].build()
                        except KeyError: # remove array_metric from dict if no records returned
                            self.experiment_timeseries_dict.pop(experiment, None)
                            warnings.warn(f'missing {sensor} records for '
                                          f'{experiment} experiment')
                        
                self.make_figures(sensor)
                        
    def make_figures(self, sensor, ncols=2):
        output_dir = os.path.join(self.config_dict['output_path'], f"{sensor}")

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
                fig, axes = plt.subplots(len(sat_set), ncols, sharex=True,sharey=True,
                                         squeeze=False)
                fig.suptitle(f"{sensor} channel {channel_num} global mean (left) and RMS (right) error (o - b)")
                #axes[-1, 0].set_xlabel = 'cycle date (Gregorian)'
                #axes[-1, 1].set_xlabel = 'cycle date (Gregorian)'
            
                for row, sat_sensor in enumerate(sorted(sat_set)):
                    sat_short_name = sat_sensor.split('_')[:-1][0]
                    sat_label = satellite_names.get_longname(sat_short_name)
                    axes[row, 0].set_title(f"{sat_label} mean error")
                    axes[row, 1].set_title(f"{sat_label} RMS error")
                    axes[row, 0].set_ylabel('temperature (K)')
                    axes[row, 0].axhline(color='black', lw=0.75)
                    axes[row, 1].axhline(color='black', lw=0.75)
                
                    experiment_idx = 0
                    for experiment, timeseries_dict in self.experiment_timeseries_dict.items():
                        yerrs=list()
                        yerrs2=list()
                        for full_stat_name, timeseries_data in timeseries_dict.items():
                            for stat_label, value_dict in timeseries_data.value_dict.items():
                                if stat_label == 'bias_post_corr_GSIstage_1' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    bias_timestamps = timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    bias_values = np.array(value_dict[sat_sensor])[:,channel_idx]
                                    std_timestamps = timeseries_dict[
                                        f'{sensor}_std_GSIstage_1'].timestamp_dict[
                                            'std_GSIstage_1'][sat_sensor]
                                    std_values = timeseries_dict[
                                        f'{sensor}_std_GSIstage_1'].value_dict[
                                            'std_GSIstage_1'][sat_sensor]
                                        
                                    for time_idx, bias_timestamp in enumerate(bias_timestamps):
                                        if bias_timestamp in std_timestamps:
                                            std_time_idx = std_timestamps.index(bias_timestamp)
                                            yerr = np.array(std_values)[std_time_idx, channel_idx]
                                            if yerr:
                                                yerrs.append(yerr)
                                            else:
                                                yerrs.append(0)
                                        else:
                                            yerrs.append(0)
                                
                                    yerrs_plot = np.array([np.nan if x is None else float(x) for x in yerrs])
                                    mean_values_plot = np.array([np.nan if x is None else float(x) for x in bias_values])
                                        
                                    axes[row, 0].bar(
                                        bias_timestamps,
                                        2.*yerrs_plot,
                                        width=pd.Timedelta(hours=6),
                                        bottom=mean_values_plot - yerrs_plot,
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=0.2
                                    )
                                
                                    axes[row, 0].errorbar(
                                        bias_timestamps,
                                        mean_values_plot,
                                        xerr=pd.Timedelta(hours=3),
                                        fmt=self.config_dict['ls_list'][experiment_idx],
                                        lw=self.config_dict['lw_list'][experiment_idx],
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha = 0.5,
                                    )
                            
                                elif stat_label == 'sqrt_bias_GSIstage_1' and sat_sensor in timeseries_data.timestamp_dict[stat_label].keys():
                                    rmse_timestamps = timeseries_data.timestamp_dict[stat_label][sat_sensor]
                                    rmse_values = np.array(value_dict[sat_sensor])[:,channel_idx]
                                    obs_err_var_timestamps = timeseries_dict[
                                        f'{sensor}_variance_GSIstage_1'].timestamp_dict[
                                            'variance_GSIstage_1'][sat_sensor]
                                    obs_err_var_values = timeseries_dict[
                                        f'{sensor}_variance_GSIstage_1'].value_dict[
                                            'variance_GSIstage_1'][sat_sensor]
                                        
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
                                
                                    yerrs_plot = np.sqrt(
                                        np.array(
                                            [0 if x is None else float(x) for x in yerrs2])
                                        )
                                    rmse_values_plot = np.array([np.nan if x is None else float(x) for x in rmse_values])
                                
                                    axes[row, 1].bar(
                                        rmse_timestamps,
                                        2.*yerrs_plot,
                                        width=pd.Timedelta(hours=6),
                                        bottom=rmse_values_plot - yerrs_plot,
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha=0.2
                                    )
                                
                                    axes[row, 1].errorbar(
                                        rmse_timestamps,
                                        rmse_values_plot,
                                        xerr=pd.Timedelta(hours=3),
                                        fmt=self.config_dict['ls_list'][experiment_idx],
                                        lw=self.config_dict['lw_list'][experiment_idx],
                                        color=self.config_dict['color_list'][experiment_idx],
                                        alpha = 0.5,
                                        label=self.friendly_names_dict[experiment]
                                    )
                                
                                    axes[row,1].legend(loc=4)
                            
                                axes[row, 0].xaxis.set_major_formatter(
                                    mdates.ConciseDateFormatter(
                                              axes[row, 0].xaxis.get_major_locator()))
                                axes[row, 1].xaxis.set_major_formatter(
                                    mdates.ConciseDateFormatter(
                                              axes[row, 1].xaxis.get_major_locator()))
                                          
                        experiment_idx += 1
                
                plt.savefig(os.path.join(self.config_dict['output_path'], 
                                  f'gsi_radiance_omb_{sensor}_ch{channel_num}.png'),
                            dpi=600)
                plt.close()

def run():
    # Initialize MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes

    # Load global configurations and friendly names
    global_config_dict, global_friendly_names_dict = config()
    sensor_list = list()
    for sensor in global_config_dict['sensor_list']:
        sensor_list.append(sensor)
    
    # Calculate how many sensors each process should handle
    sensors_per_process = len(sensor_list) // size

    # Handle leftover sensors (remaining sensors are distributed to the first few processes)
    leftover_sensors = len(sensor_list) % size

    # Calculate the start and end indices for each process
    start_idx = rank * sensors_per_process + min(rank, leftover_sensors)  # Adjust start index for extra sensors
    end_idx = start_idx + sensors_per_process + (1 if rank < leftover_sensors else 0)  # Adjust end index for extra sensors

    # Slice the sensor list for this process
    local_sensor_list = sensor_list[start_idx:end_idx]

    # Create an instance of GSIRadianceFit2ObsFig for experiment data
    experiment_metrics_timeseries_data = GSIRadianceFit2ObsFig()
    
    # Each process handles its portion of the sensor list
    for sensor in local_sensor_list:
        # Set the current sensor for the experiment
        experiment_metrics_timeseries_data.config_dict['sensor_list'] = [sensor]
        
        #print(f'{sensor}', rank)
        
        # Build time series data for the current sensor
        experiment_metrics_timeseries_data.build_timeseries()


def main():
    """
    """
    run()

if __name__ == "__main__":
    main()