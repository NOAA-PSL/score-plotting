#!/usr/bin/env python

"""build GSI stats time series for plotting

usage: place this file in score-db/src, run using the python interpreter, i.e.

score-db/src$ python gsistats_timeseries.py

users shouldn't need to edit anything besides the run() function, which can
be customized according to their needs

This script uses a matplotlib styesheet. To make the style sheet available to
matplotlib, place the "agu_full.mplstyle" file in the "stylelib" direcotry under matplotlib.get_configdir(), which is usually either ~/.config/matplotlib/ or ~/.matplotlib/ (https://matplotlib.org/stable/users/explain/customizing.html#using-style-sheets)

for any questions, please feel free to contact Adam Schneider 
(Adam.Schneider@noaa.gov)
"""

import os
import pathlib
from datetime import datetime
import warnings

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import colorcet as cc

from score_db import score_db_base

CONFIG_PATH = os.path.join(
        pathlib.Path(__file__).parent.parent.parent.resolve(),
        'conf'
)
CONFIG_FILE = 'agu_full.mplstyle'

friendly_names_dict={"scout_run_v1":"NOAA Scout Run", "NASA_GEOSIT_GSISTATS":"NASA GEOS-IT", "std_GSIstage_1":"STD", "bias_post_corr_GSIstage_1":"Bias"}

def run(make_plot=False, make_line_plot=True, select_array_metric_types=True,
        select_sat_name=True,
        experiment_list=['scout_run_v1',
                         'NASA_GEOSIT_GSISTATS'
                         #'scout_runs_gsi3dvar_1979stream'fcdvwq
                     ],
        array_metrics_list=['amsua_bias_post_corr_GSIstage_%',
                            #'amsua_std_%',
                            #'%_variance_%',
                            #'amsua_use_%'
                        ],
        sat_name = 'NOAA 15',
        start_date = '1979-01-01 00:00:00',
        stop_date = '2001-06-01 00:00:00'):
    """modify the above input variables to configure and generate time series
    data for various GSI related statistics
        
        experiment_list: list of experiments to plot in sequence
        array_metrics_list: list of metrics to plot in sequence
    """

    style_file = os.path.join(
            CONFIG_PATH,
            CONFIG_FILE
    )
    if make_plot or make_line_plot:
        plt.style.use(style_file)

    if not select_array_metric_types:
        array_metrics_list=['%_all_stats_%']
    
    for experiment_name in experiment_list:
        for array_metric_type in array_metrics_list:
            timeseries_data = GSIStatsTimeSeries(start_date, stop_date,
                            experiment_name=experiment_name, 
                            select_array_metric_types=select_array_metric_types,
                            array_metric_types=array_metric_type,
                            select_sat_name=select_sat_name,
                            sat_name=sat_name)
            timeseries_data.build(all_channel_max=False, # set max or mean
                                  all_channel_mean=False,
                                  by_channel=True) # other False

            if make_plot:
                timeseries_data.plot()
                plt.suptitle(experiment_name)
                #plt.show()
                metric_string = array_metric_type.split('%')[1] #this won't always work if you give a specific sensor value
                plt.savefig(os.path.join(
                                'results',
                                f'gsi{metric_string}{experiment_name}.png'),
                    dpi=600)
                plt.close()

            elif make_line_plot:
                stat_label = 'bias_post_corr_GSIstage_1'
                #stat_label = 'std_GSIstage_1'
                sensor_label = 'n15_amsua'
                y_min = -0.5
                y_max = 0.6
                timeseries_data.plot_line_plot(stat_label=stat_label, sensor_label=sensor_label, experiment_name=experiment_name, y_min=y_min, y_max=y_max)
                metric_string = array_metric_type.split('%')[0] #again not expandable 
                plt.savefig(os.path.join(
                                #'results',
                                f'gsiline{metric_string}{experiment_name}.png'),
                    dpi=600)
                plt.close()
            else:
                timeseries_data.print_init_time()

# separate to be able to plot experiments on the same graphic / flattens data for now
def run_line_plot(make_line_plot=True, select_array_metric_types=True,
        select_sat_name=True, multi_stat=False,
        experiment_list=[#'scout_run_v1',
                         'NASA_GEOSIT_GSISTATS',
                         'scout_run_v1'
                         #'scout_runs_gsi3dvar_1979stream'
                     ],
        array_metrics_list=[#'amsua_std_%',
                            #'amsua_bias_post_corr_GSIstage_%',
                            #'%_variance_%',
                            'amsua_nobs_used_%'
                        ],
        sat_name = 'NOAA 15',
        channel_indices = [4, 5, 6, 7], #this is the specific location in the array, not based on the channel name that needs to be expanded
        start_date = '1999-01-01 00:00:00',
        stop_date = '2001-06-01 00:00:00'):
    """modify the above input variables to configure and generate time series
    data for various GSI related statistics
        
        experiment_list: list of experiments to plot in sequence
        array_metrics_list: list of metrics to plot in sequence
    """
    
    style_file = os.path.join(
            CONFIG_PATH,
            CONFIG_FILE
    )
    if make_plot or make_line_plot:
        plt.style.use(style_file)
        plt.rcParams['font.size'] = 20

    if not select_array_metric_types:
        array_metrics_list=['%_all_stats_%']

    if multi_stat:
        experiment_timeseries = dict()

        for experiment_name in experiment_list:
            experiment_timeseries[experiment_name] = dict()  # Create a dictionary for each experiment
            for array_metric_type in array_metrics_list:
                timeseries_data = GSIStatsTimeSeries(
                                    start_date, stop_date,
                                    experiment_name=experiment_name, 
                                    select_array_metric_types=select_array_metric_types,
                                    array_metric_types=array_metric_type,
                                    select_sat_name=select_sat_name,
                                    sat_name=sat_name)
                
                # Flatten data for the selected channels
                timeseries_data.flatten_by_channel(channel_indices=channel_indices)
                
                # Store the timeseries data by experiment name and array metric type
                experiment_timeseries[experiment_name][array_metric_type] = timeseries_data
    else:
        experiment_timeseries = dict()
        for experiment_name in experiment_list:
            for array_metric_type in array_metrics_list:
                timeseries_data = GSIStatsTimeSeries(start_date, stop_date,
                                experiment_name=experiment_name, 
                                select_array_metric_types=select_array_metric_types,
                                array_metric_types=array_metric_type,
                                select_sat_name=select_sat_name,
                                sat_name=sat_name)
                timeseries_data.flatten_by_channel(channel_indices=channel_indices)
                #timeseries_data.build(by_channel=True)
                experiment_timeseries[experiment_name] = timeseries_data

    if make_line_plot:
        # stat_label = 'amsua_bias_post_corr_GSIstage_1'
        # sensor_label = 'n15_amsua'

        plot_experiment_comparison(experiment_timeseries, experiment_list, ".", "5, 6, 7, 8", ['#E4002B', '#003087'], 0)

        #plot_experiment_comparison_multi_stat(experiment_timeseries, experiment_list, ".", "8", ['std_GSIstage_1', 'bias_post_corr_GSIstage_1'], array_metrics_list, [['#003087', '#0085CA'], ['#E4002B', '#f2901f']], -0.2, 0.4)
        
        #plot_experiment_comparison_by_channel(experiment_timeseries, experiment_list, ".", channel_indices)
    else:
        timeseries_data.print_init_time()

class GSIStatsTimeSeries(object):
    def __init__(self, start_date, stop_date,
                 experiment_name = 'scout_run_v1',#
                                   #'scout_runs_gsi3dvar_1979stream',#
                select_array_metric_types = True,
                array_metric_types='%',
                select_sat_name = False,
                sat_name = None
                ):
        """Download metrics data for given experiment name
        """
        self.init_datetime = datetime.now()
        self.experiment_name = experiment_name
        self.select_array_metric_types = select_array_metric_types
        self.array_metric_types = array_metric_types
        self.select_sat_name = select_sat_name
        self.sat_name = sat_name
        self.get_data_frame(start_date, stop_date)

        
    def get_data_frame(self, start_date, stop_date):
        """request from the score-db application experiment data
        Database requests are submitted via score-db with a request dictionary
        """
        request_dict = {
            'db_request_name': 'expt_array_metrics',
            'method': 'GET',
            'params': {'filters':
                          {'experiment':{
                              'experiment_name':
                                  {'exact':
                                     self.experiment_name}
                                 },
                           'regions': {
                                            'name': {
                                                'exact': ['global']
                                            },
                                        },

                           'time_valid': {
                                            'from': start_date,
                                            'to': stop_date,
                                        },
                                    },
                       'ordering': [ {'name': 'time_valid', 'order_by': 'asc'}]
                   }
        
        }
    
        if self.select_array_metric_types:
            request_dict['params']['filters']['array_metric_types'] = {
                'name': {'like': self.array_metric_types}
            }

        if self.select_sat_name:
            request_dict['params']['filters']['sat_meta'] = {
                'sat_name': {'exact': self.sat_name}
            }

        db_action_response = score_db_base.handle_request(request_dict)    
        self.data_frame = db_action_response.details['records']
        
        # sort by timestamp, created at
        self.data_frame.sort_values(by=['metric_instrument_name',
                                        'sat_short_name',
                                        'time_valid',
                                        'created_at'], 
                                    inplace=True)
    
        # remove duplicate data
        self.data_frame.drop_duplicates(subset=['metric_name', 'time_valid'], 
                                        keep='last', inplace=True)
        
    def build(self, all_channel_max=True, all_channel_mean=False, by_channel=False):
        self.unique_stat_list = extract_unique_stats(
                                            set(self.data_frame['metric_name']))
        
        self.timestamp_dict = dict()
        self.timelabel_dict = dict()
        self.value_dict = dict()
        for i, stat_name in enumerate(self.unique_stat_list[0]):
            for j, gsi_stage in enumerate(self.unique_stat_list[1]):
                self.timestamp_dict[f'{stat_name}_GSIstage_{gsi_stage}'] = dict()
                self.timelabel_dict[f'{stat_name}_GSIstage_{gsi_stage}'] = dict()
                self.value_dict[f'{stat_name}_GSIstage_{gsi_stage}'] = dict()
                
        
        for key in self.value_dict.keys():
            for sat_short_name in set(self.data_frame.sat_short_name):
                for instrument_name in set(
                                    self.data_frame.metric_instrument_name):
                    sensor_label = f'{sat_short_name}_{instrument_name}'
                    
                    self.timestamp_dict[key][sensor_label] = list()
                    self.timelabel_dict[key][sensor_label] = list()
                    self.value_dict[key][sensor_label] = list()
        
        self.sensorlabel_dict = dict()
        yval = 0
        for row in self.data_frame.itertuples():
            metric_name_parts = row.metric_name.split('_')

            if metric_name_parts[0] == row.metric_instrument_name and metric_name_parts[-1] != 'None':
                stat_name = '_'.join(metric_name_parts[1:-2])
                gsi_stage = metric_name_parts[-1]
                
                stat_label = f'{stat_name}_GSIstage_{gsi_stage}'
                
                sensor_label = f'{row.sat_short_name}_{row.metric_instrument_name}'
                timestamp = row.time_valid#.timestamp()
                time_label = '%02d-%02d-%04d' % (row.time_valid.month,
                                             row.time_valid.day,
                                             row.time_valid.year,)
                
                if all_channel_mean and all_channel_max:
                    warnings.warn("got both channel mean and max, returning "
                                  "mean")
                    value = np.mean(row.value)
                elif all_channel_mean:
                    try:
                        value = np.mean(row.value)
                    except TypeError:
                        value = np.nan
                elif all_channel_max:
                    try:
                        value = np.max(row.value)
                    except TypeError:
                        value = np.nan
                elif by_channel: 
                    value = row.value

                
                #print(gsi_stage, stat_name, sensor_label, time_label, value)
                self.timestamp_dict[stat_label][sensor_label].append(timestamp)
                self.timelabel_dict[stat_label][sensor_label].append(time_label)
                self.value_dict[stat_label][sensor_label].append(value)
                
                if not sensor_label in self.sensorlabel_dict.keys():
                    self.sensorlabel_dict[sensor_label] = yval
                    yval -= 1
        
                #print(gsi_stage, stat_name, sensor_label, self.sensorlabel_dict[sensor_label])

    def flatten(self):
        self.unique_stat_list = extract_unique_stats(
                                            set(self.data_frame['metric_name']))
        
        self.timestamp_dict = dict()
        self.timelabel_dict = dict()
        self.value_dict = dict()
        
        
        self.sensorlabel_list = list()
        self.statlabel_list = list()
        yval = 0
        for row in self.data_frame.itertuples():
            metric_name_parts = row.metric_name.split('_')

            if metric_name_parts[0] == row.metric_instrument_name and metric_name_parts[-1] != 'None':
                stat_name = '_'.join(metric_name_parts[1:-2])
                gsi_stage = metric_name_parts[-1]
                
                stat_label = f'{stat_name}_GSIstage_{gsi_stage}'
                
                sensor_label = f'{row.sat_short_name}_{row.metric_instrument_name}'
                timestamp = row.time_valid#.timestamp()
                time_label = '%02d-%02d-%04d' % (row.time_valid.month,
                                             row.time_valid.day,
                                             row.time_valid.year,)
                
                value = np.nansum([np.nan if v is None else v for v in row.value]) #flatten to one total

                # Check if stat_label exists in timestamp_dict
                if stat_label not in self.timestamp_dict:
                    self.timestamp_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                # Check if sensor_label exists under stat_label in timestamp_dict
                if sensor_label not in self.timestamp_dict[stat_label]:
                    self.timestamp_dict[stat_label][sensor_label] = []  # Create an empty list for sensor_label

                # Check if stat_label exists in timelabel_dict
                if stat_label not in self.timelabel_dict:
                    self.timelabel_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                # Check if sensor_label exists under stat_label in timelabel_dict
                if sensor_label not in self.timelabel_dict[stat_label]:
                    self.timelabel_dict[stat_label][sensor_label] = []  # Create an empty list for sensor_label
                
                # Check if stat_label exists in timelabel_dict
                if stat_label not in self.value_dict:
                    self.value_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                # Check if sensor_label exists under stat_label in timelabel_dict
                if sensor_label not in self.value_dict[stat_label]:
                    self.value_dict[stat_label][sensor_label] = []  # Create an empty list for sensor_label

                #print(gsi_stage, stat_name, sensor_label, time_label, value)
                self.timestamp_dict[stat_label][sensor_label].append(timestamp)
                self.timelabel_dict[stat_label][sensor_label].append(time_label)
                self.value_dict[stat_label][sensor_label].append(value)
                
                if not sensor_label in self.sensorlabel_list:
                    self.sensorlabel_list.append(sensor_label)
                
                if not stat_label in self.statlabel_list:
                    self.statlabel_list.append(stat_label)
        
                #print(gsi_stage, stat_name, sensor_label, self.sensorlabel_dict[sensor_label])

    #right now this function just selects by the channel indices but it should be expanded to use channel names and then applied to the value indices
    def flatten_by_channel(self, channel_indices):
        if channel_indices is None:
            self.flatten() #do a basic full flatten instead
            return

        self.unique_stat_list = extract_unique_stats(
                                            set(self.data_frame['metric_name']))
        
        self.timestamp_dict = dict()
        self.timelabel_dict = dict()
        self.value_dict = dict()
        
        
        self.sensorlabel_list = list()
        self.statlabel_list = list()
        yval = 0
        for row in self.data_frame.itertuples():
            metric_name_parts = row.metric_name.split('_')

            if metric_name_parts[0] == row.metric_instrument_name and metric_name_parts[-1] != 'None':
                stat_name = '_'.join(metric_name_parts[1:-2])
                gsi_stage = metric_name_parts[-1]
                
                stat_label = f'{stat_name}_GSIstage_{gsi_stage}'
                
                sensor_label = f'{row.sat_short_name}_{row.metric_instrument_name}'
                timestamp = row.time_valid#.timestamp()
                time_label = '%02d-%02d-%04d' % (row.time_valid.month,
                                             row.time_valid.day,
                                             row.time_valid.year,)
                
                #flatten to only include given channels
                value = np.nansum([np.nan if row.value[i] is None else row.value[i] for i in channel_indices if i < len(row.value)])

                # Check if stat_label exists in timestamp_dict
                if stat_label not in self.timestamp_dict:
                    self.timestamp_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                # Check if sensor_label exists under stat_label in timestamp_dict
                if sensor_label not in self.timestamp_dict[stat_label]:
                    self.timestamp_dict[stat_label][sensor_label] = []  # Create an empty list for sensor_label

                # Check if stat_label exists in timelabel_dict
                if stat_label not in self.timelabel_dict:
                    self.timelabel_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                # Check if sensor_label exists under stat_label in timelabel_dict
                if sensor_label not in self.timelabel_dict[stat_label]:
                    self.timelabel_dict[stat_label][sensor_label] = []  # Create an empty list for sensor_label
                
                # Check if stat_label exists in timelabel_dict
                if stat_label not in self.value_dict:
                    self.value_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                # Check if sensor_label exists under stat_label in timelabel_dict
                if sensor_label not in self.value_dict[stat_label]:
                    self.value_dict[stat_label][sensor_label] = []  # Create an empty list for sensor_label

                #print(gsi_stage, stat_name, sensor_label, time_label, value)
                self.timestamp_dict[stat_label][sensor_label].append(timestamp)
                self.timelabel_dict[stat_label][sensor_label].append(time_label)
                self.value_dict[stat_label][sensor_label].append(value)
                
                if not sensor_label in self.sensorlabel_list:
                    self.sensorlabel_list.append(sensor_label)
                
                if not stat_label in self.statlabel_list:
                    self.statlabel_list.append(stat_label)
        
    def plot(self, all_channel_mean=False, all_channel_max=True):
        """demonstrate how to plot metrics stored in a backened SQL database
        """
        cmap, boundaries, norm, tick_positions = get_colormap()
        fig, axes = plt.subplots(nrows = len(self.unique_stat_list[0]),
                                 ncols = len(self.unique_stat_list[1]),
                                 sharex=True, sharey=True,
                                 squeeze=False)
        
        ylabels = list()
        yvals = list()
        for sensor_label, yval in self.sensorlabel_dict.items():
            ylabels.append(sensor_label)
            yvals.append(yval)
                
        for row, stat_name in enumerate(self.unique_stat_list[0]):
            for col, gsi_stage in enumerate(self.unique_stat_list[1]):
                stat_label = f'{stat_name}_GSIstage_{gsi_stage}'
                
                if all_channel_mean:
                    axes[row, col].set_title(f'all channel mean {stat_name} (GSI stage {gsi_stage})')
                elif all_channel_max:
                    axes[row, col].set_title(f'all channel max {stat_name} (GSI stage {gsi_stage})')
                
                # y labels
                axes[row, col].set_yticks(np.array(yvals) - 0.5, 
                                          labels=ylabels, rotation=30, 
                                               va='center_baseline')
                axes[row, col].set_yticks(np.array(yvals), minor=True)
                axes[row, col].set_ylim(-len(yvals), 0)
                
                axes[row, col].grid(color='black', alpha=0.1, which='minor')
                
                # color mesh
                for sensor_label, yval in self.sensorlabel_dict.items():
                    values = self.value_dict[stat_label][sensor_label]
                    
                    # time dimension
                    timestamps = self.timestamp_dict[stat_label][sensor_label]
                    
                    try:
                        timestamps.append(timestamps[-1])
                        cax = axes[row, col].pcolormesh(np.array(timestamps),
                                                    np.array([yval, yval - 1]),
                                                    np.array([values]),
                                                    cmap=cmap,
                                                    norm=norm,
                                                    shading='flat')
                                                    
                        axes[row, col].xaxis.set_major_formatter(
                            mdates.ConciseDateFormatter(
                                      axes[row, col].xaxis.get_major_locator()))
                        axes[row, col].tick_params(which='major', labeltop=True,
                                                   labelright=False,
                                           top=True, right=False)
                        axes[row, col].tick_params(which='minor', left=False,
                                                   bottom=True, right=False,
                                                   top=True)
                
                    except IndexError:
                        warnings.warn(f'no data to plot for {stat_label} {sensor_label}')
                
                '''
                
                
                # Major ticks every half year, minor ticks every month,
                axes[row, col].xaxis.set_major_locator(
                                            mdates.MonthLocator(bymonth=(1, 7)))
                
                
                
                axes[row, col].xaxis.set_minor_locator(mdates.MonthLocator())
                axes[row, col].set_xlabel('cycle date (Gregorian)')
                for label in axes[row, col].get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')
                '''
            
        # Add a colorbar to the plot with the same limits
        cbar = fig.colorbar(cax, ax=axes, orientation='horizontal',
                            #pad=0.1
                            boundaries=boundaries)
        cbar.set_label('temperature (K)')
        cbar.set_ticks(tick_positions)
        tick_labels = [f'{pos:.1f}' for pos in tick_positions]
        cbar.set_ticklabels(tick_labels)
    
    def print_init_time(self):
        print("GSIStatsTimeSeries object init date and time: ",
              f"{self.init_datetime}") 

    def plot_line_plot(self, stat_label, sensor_label, experiment_name, channels_to_plot=[4, 5, 6, 7], y_min=None, y_max=None):
        """
        Plot time series for specified stat_label, sensor_label, and channels.
        
        Parameters:
        - stat_label: The specific stat label (string) to plot.
        - sensor_label: The specific sensor label (string) to plot.
        - channels_to_plot: List of channel indices to plot (default: channels 5-8). 
                            -1 for indexing, should read from the array labels in the future.
        - y_min: Minimum limit for the y-axis (optional).
        - y_max: Maximum limit for the y-axis (optional).
        """
        
        # Prepare the plot
        plt.figure(figsize=(12, 8), dpi=300)

        # Loop through the specified channels
        for channel in channels_to_plot:
            try:
                # Extract the values for the specified stat_label and sensor_label
                channel_values = self.value_dict[stat_label][sensor_label]

                # Extract the corresponding timestamps
                timestamps = self.timestamp_dict[stat_label][sensor_label]

                # Ensure the channel index is valid and plot the values
                if len(channel_values) > channel:
                    plt.plot(timestamps, [v[channel] for v in channel_values], label=f'Channel {channel + 1}', alpha=0.7)
                else:
                    print(f"Channel {channel + 1} not found for {stat_label}, {sensor_label}")
                    
            except KeyError as e:
                print(f"Missing data for {stat_label}, {sensor_label}: {e}")

        # Add labels and title
        plt.xlabel('Timestamp')
        plt.ylabel(f'{stat_label}')
        plt.title(f'{experiment_name} Channel Values vs Time for {sensor_label} ({stat_label})')
        plt.legend()

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)

        # Set y-axis limits if specified
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        elif y_max is not None:
            plt.ylim(top=y_max)

        # Adjust layout to avoid label clipping
        plt.tight_layout()




def get_colormap(cmap = cc.cm.CET_D1A, discrete_levels = 51, num_ticks=11,
                 vmin=-5, vmax=5):
    # Create discrete colormap
    colors = cmap(np.linspace(0, 1, discrete_levels))
    cmap_discrete = mcolors.ListedColormap(colors)

    # Create boundaries for the colormap
    boundaries = np.linspace(vmin, vmax, discrete_levels)
    norm = mcolors.BoundaryNorm(boundaries, cmap_discrete.N)
    
    # Adjust tick labels: Use fewer ticks
    tick_positions = np.linspace(vmin, vmax, num_ticks)
    
    return(cmap_discrete, boundaries, norm, tick_positions)
    
def extract_unique_stats(strings):
    # Create sets to store unique values in the second and last positions
    second_position_set = set()
    last_position_set = set()
    
    # Iterate over the set of strings
    for s in strings:
        parts = s.split('_')
        
        # Add the second and last elements to their respective sets
        if len(parts) > 1 and parts[-1] != 'None':  # Ensure there are at least 2 parts, 
            second_position_set.add('_'.join(parts[1:-2]))
            last_position_set.add(parts[-1])
    
    # Convert sets to sorted lists to maintain a consistent order
    second_position_list = sorted(list(second_position_set))
    last_position_list = sorted(list(last_position_set))
    
    # Combine the two lists into a 2D array (list of lists)
    unique_positions = [second_position_list, last_position_list]
    
    return unique_positions

def make_line_plot_multi_expt(timeseries_dict, experiment_list):
    """
    Plot time series for multiple experiments stored in a dictionary.
    
    Parameters:
    - timeseries_dict: Dictionary where keys are experiment names and values are GSIStatsTimeSeries objects.
    - experiment_list: List of experiment names to plot.
    """
    
    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # Loop through each experiment name in experiment_list
    for experiment_name in experiment_list:
        # Access the corresponding GSIStatsTimeSeries object from the dictionary
        timeseries_obj = timeseries_dict.get(experiment_name)

        if timeseries_obj:
            # Extract data from the timeseries object
            time_valid = timeseries_obj.timestamp_dict  # Replace with actual column name
            value = timeseries_obj.value_dict         # Replace with actual column name


            # Plot the data
            plt.plot(time_valid, value, label=experiment_name)
        else:
            print(f"No data found for experiment: {experiment_name}")
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Number AMSUA Obs Used')
    plt.title(f'Time Series Comparison of Experiments')
    plt.legend()

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_experiment_comparison(timeseries_dict, experiment_list, output_dir, channel_list, expt_colors=None, y_min=None, y_max=None):
    """
    Plot time series for multiple experiments for multiple stat and sensor combination, and save each plot.
    
    Parameters:
    - timeseries_dict: Dictionary where keys are experiment names and values are GSIStatsTimeSeries objects.
    - experiment_list: List of experiment names to plot.
    - output_dir: Directory where plots will be saved.
    """
    
    # Get statlabel_list and sensorlabel_list from one of the GSIStatsTimeSeries objects
    if not timeseries_dict:
        print("Error: timeseries_dict is empty.")
        return
    
    # Extract the statlabel_list and sensorlabel_list from the first object in timeseries_dict
    first_timeseries_obj = list(timeseries_dict.values())[0]
    statlabel_list = first_timeseries_obj.statlabel_list
    sensorlabel_list = first_timeseries_obj.sensorlabel_list

    # Loop through each stat_label in the statlabel_list
    for stat_label in statlabel_list:
        # Loop through each sensor_label in the sensorlabel_list
        for sensor_label in sensorlabel_list:
            plt.figure(figsize=(16, 12), dpi=300)  # Create a new figure for each stat-sensor combination

            # Loop through each experiment in the experiment list
            for i, experiment_name in enumerate(experiment_list):
                # Access the corresponding GSIStatsTimeSeries object from the dictionary
                timeseries_obj = timeseries_dict.get(experiment_name)

                if timeseries_obj:
                    # Safely access the nested dictionary for time_valid and value
                    time_valid = timeseries_obj.timestamp_dict.get(stat_label, {}).get(sensor_label, [])
                    value = timeseries_obj.value_dict.get(stat_label, {}).get(sensor_label, [])

                    # Check if data exists for the stat_label and sensor_label
                    if time_valid and value:
                        # Plot the data for this experiment
                        color = expt_colors[i] if expt_colors else None
                        experiment_label = experiment_name
                        if experiment_name in friendly_names_dict:
                            experiment_label = friendly_names_dict[experiment_name]
                        plt.plot(time_valid, value, label=experiment_label, alpha=0.6, color=color) #set plot for line or bar for bar or scatter for scatter
                    else:
                        print(f"No data for {stat_label}, {sensor_label} in experiment: {experiment_name}")
                else:
                    print(f"No data for experiment: {experiment_name}")

            # Set y-axis limits if specified
            if y_min is not None or y_max is not None:
                plt.ylim(y_min, y_max)

            # Add labels and title for the plot
            plt.xlabel('Time Valid', fontsize=18)
            plt.ylabel(f'{stat_label}', fontsize=18)
            if channel_list is None:
                plt.title(f'Comparison of {stat_label} and {sensor_label} across Experiments', fontsize=18)
            else:
                plt.title(f'Comparison of {stat_label} and {sensor_label} across Experiments for Channels {channel_list}', fontsize=18)
            plt.legend(fontsize=20)

            # Rotate x-axis labels for readability
            plt.xticks(rotation=45, fontsize=20)
            plt.yticks(fontsize=20)

            # Save the plot to the specified output directory
            plot_filename = f'{stat_label}_{sensor_label}_comparison.png'
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath)

            # Close the plot after saving
            plt.close()

            print(f"Plot saved: {plot_filepath}")

def plot_experiment_comparison_multi_stat(timeseries_dict, experiment_list, output_dir, channel_list, stat_pair, array_metrics_list, line_colors=None, y_min=None, y_max=None):
    """
    Plot time series for multiple experiments for each stat and sensor combination, and save each plot.

    Parameters:
    - timeseries_dict: Dictionary where keys are experiment names and values are nested dictionaries of array metric types.
    - experiment_list: List of experiment names to plot.
    - output_dir: Directory where plots will be saved.
    - channel_list: List of channel indices to plot.
    - stat_pair: List containing two stat labels to compare (e.g., ['std_GSIstage_1', 'bias_post_corr_GSIstage_1']).
    - array_metrics_list: List of array metric types corresponding to the stat labels.
    - line_colors: List of colors for each line (optional). If None, default colors will be used.
    """

    # Get statlabel_list and sensorlabel_list from one of the GSIStatsTimeSeries objects
    if not timeseries_dict:
        print("Error: timeseries_dict is empty.")
        return

    # Extract the statlabel_list and sensorlabel_list from the first object in timeseries_dict
    first_timeseries_obj = list(timeseries_dict.values())[0].values()
    sensorlabel_list = list(first_timeseries_obj)[0].sensorlabel_list

    # Set default line colors if none are provided
    # if line_colors is None:
    #     line_colors = plt.cm.get_cmap('tab10', len(experiment_list))

    # Loop through each sensor_label in the sensorlabel_list
    for sensor_label in sensorlabel_list:
        plt.figure(figsize=(16, 12), dpi=300)  # Create a new figure for each sensor combination

        # Loop through each experiment in the experiment list
        for i, experiment_name in enumerate(experiment_list):
            # Access the corresponding dictionary for the experiment
            experiment_data = timeseries_dict.get(experiment_name)

            if experiment_data:
                # Loop through both stat labels in the stat_pair
                for j, stat_label in enumerate(stat_pair):
                    array_metric_type = array_metrics_list[j]  # Map stat_label to array_metric_type
                    timeseries_obj = experiment_data.get(array_metric_type)

                    if timeseries_obj:
                        # Safely access the nested dictionary for time_valid and value
                        time_valid = timeseries_obj.timestamp_dict.get(stat_label, {}).get(sensor_label, [])
                        value = timeseries_obj.value_dict.get(stat_label, {}).get(sensor_label, [])

                        # Check if data exists for the stat_label and sensor_label
                        if time_valid and value:
                            # Plot the data for this experiment and stat_label with custom color
                            color = line_colors[i][j] if line_colors else None
                            experiment_label = experiment_name
                            stat_friendly = stat_label
                            if experiment_name in friendly_names_dict:
                                experiment_label = friendly_names_dict[experiment_name]
                            if stat_label in friendly_names_dict:
                                stat_friendly = friendly_names_dict[stat_label]
                            plt.plot(time_valid, value, label=f'{experiment_label} - {stat_friendly}', color=color, alpha=0.6)
                        else:
                            print(f"No data for {stat_label}, {sensor_label} in experiment: {experiment_name}")
                    else:
                        print(f"No data for array metric type {array_metric_type} in experiment: {experiment_name}")
            else:
                print(f"No data for experiment: {experiment_name}")

        # Set y-axis limits if specified
        if y_min is not None or y_max is not None:
            plt.ylim(y_min, y_max)

        # Add labels and title for the plot
        plt.xlabel('Time Valid', fontsize=18)
        plt.ylabel(f'Statistic Values', fontsize=18)
        plt.title(f'Comparison of NOAA and NASA Experiments for Channel {channel_list}', fontsize=18)
        plt.legend(fontsize=20)

        #Rotate x-axis labels for readability
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)

        # Save the plot to the specified output directory
        plot_filename = f'{sensor_label}_comparison_{stat_pair[0]}_{stat_pair[1]}.png'
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath)

        # Close the plot after saving
        plt.close()

        print(f"Plot saved: {plot_filepath}")



#in progress, not being used yet successfully 
def plot_experiment_comparison_by_channel(timeseries_dict, experiment_list, output_dir, channels_to_plot):
    """
    Plot time series for multiple experiments for each stat and sensor combination, and save each plot.
    
    Parameters:
    - timeseries_dict: Dictionary where keys are experiment names and values are GSIStatsTimeSeries objects.
    - experiment_list: List of experiment names to plot.
    - output_dir: Directory where plots will be saved.
    """
    
    # Get statlabel_list and sensorlabel_list from one of the GSIStatsTimeSeries objects
    if not timeseries_dict:
        print("Error: timeseries_dict is empty.")
        return
    
    # Extract the statlabel_list and sensorlabel_list from the first object in timeseries_dict
    first_timeseries_obj = list(timeseries_dict.values())[0]
    statlabel_list = first_timeseries_obj.statlabel_list
    sensorlabel_list = first_timeseries_obj.sensorlabel_list

    for channel in channels_to_plot:
        # Loop through each stat_label in the statlabel_list
        for stat_label in statlabel_list:
            # Loop through each sensor_label in the sensorlabel_list
            for sensor_label in sensorlabel_list:
                plt.figure(figsize=(12, 8), dpi=300)  # Create a new figure for each stat-sensor combination

                # Loop through each experiment in the experiment list
                for experiment_name in experiment_list:
                    # Access the corresponding GSIStatsTimeSeries object from the dictionary
                    timeseries_obj = timeseries_dict.get(experiment_name)

                    if timeseries_obj:
                        # Safely access the nested dictionary for time_valid and value
                        time_valid = timeseries_obj.timestamp_dict.get(stat_label, {}).get(sensor_label, [])
                        value = timeseries_obj.value_dict.get(stat_label, {}).get(sensor_label, [])[channel]

                        # Check if data exists for the stat_label and sensor_label
                        if time_valid and value:
                            # Plot the data for this experiment
                            plt.plot(time_valid, value, label=experiment_name, alpha=0.7) #set plot for line or bar for bar or scatter for scatter
                        else:
                            print(f"No data for {stat_label}, {sensor_label} in experiment: {experiment_name}")
                    else:
                        print(f"No data for experiment: {experiment_name}")

                # Add labels and title for the plot
                plt.xlabel('Time Valid', fontsize=24)
                plt.ylabel(f'{stat_label}', fontsize=24)
                plt.title(f'Comparison of {stat_label} and {sensor_label} across Experiments for Channel {channel + 1}')
                plt.legend()

                # Rotate x-axis labels for readability
                # Rotate x-axis labels for readability
                plt.xticks(rotation=45, fontsize=24)
                plt.yticks(fontsize=24)

                # Save the plot to the specified output directory
                plot_filename = f'{stat_label}_{sensor_label}_comparison.svg'
                plot_filepath = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_filepath)

                # Close the plot after saving
                plt.close()

                print(f"Plot saved: {plot_filepath}")



def main():
    run()

if __name__=='__main__':
    main()
