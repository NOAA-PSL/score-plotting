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

def run(make_plot=False, make_line_plot=True, select_array_metric_types=True,
        select_sat_name=True,
        experiment_list=['scout_run_v1',
                         'NASA_GEOSIT_GSISTATS'
                         #'scout_runs_gsi3dvar_1979stream'
                     ],
        array_metrics_list=['amsua_bias_post_corr_GSIstage_%',
                            #'amsua_std_%',
                            #'%_variance_%',
                            #'amsua_use_%'
                        ],
        sat_name = 'NOAA 15',
        channel_list = ['5','6','7'],
        start_date = '1979-01-01 00:00:00',
        stop_date = '2026-01-01 00:00:00'):
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
                timeseries_data.flatten_by_channel(channel_list)
                timeseries_data.plot_line_plot_by_channel(stat_label=stat_label, sensor_label=sensor_label, experiment_name=experiment_name, channels_to_plot=channel_list, y_min=y_min, y_max=y_max)
                metric_string = array_metric_type.split('%')[0] #again not expandable 
                plt.savefig(os.path.join(
                                #'results',
                                f'gsiline{metric_string}{experiment_name}.png'),
                    dpi=600)
                plt.close()
            else:
                timeseries_data.print_init_time()

def run_line_plot(make_line_plot=True, select_array_metric_types=True,
        select_sat_name=True, multi_stat=True, per_channel=True,
        experiment_list=[
                         'NASA_GEOSIT_GSISTATS',
                         'scout_run_v1',
                         #'replay_observer_diagnostic_v1'
                     ],
        array_metrics_list=['amsua_std_%',
                            'amsua_bias_post_corr_GSIstage_%',
                            #'amsua_nobs_used_%'
                        ],
        sensor_name = 'AMSUA',
        sat_name = 'NOAA 18',
        channel_list = None, 
        start_date = '1999-01-01 00:00:00',
        stop_date = '2024-12-01 00:00:00',
        color_list = [['#E4002B', '#f2901f'], 
                      ['#003087', '#0085CA'],
                      #['#46990f', '#c1e67c']
                      ],
        stat_pair = ['std_GSIstage_1', 'bias_post_corr_GSIstage_1'],
        y_min = None, 
        y_max = None,
        output_directory="."):
    """modify the above input variables to configure and generate time series
    data for various GSI related statistics
        
        experiment_list: list of experiments to plot in sequence
        array_metrics_list: list of metrics to plot in sequence
    """
    
    style_file = os.path.join(
            CONFIG_PATH,
            CONFIG_FILE
    )
    if make_line_plot:
        plt.style.use(style_file)
        plt.rcParams['font.size'] = 20

    if not select_array_metric_types:
        array_metrics_list=['%_all_stats_%']

    if multi_stat:
        experiment_timeseries = dict()

        #for experiment_name in experiment_list:
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
                timeseries_data.flatten_by_channel(channel_list=channel_list)
                
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
                timeseries_data.flatten_by_channel(channel_list=channel_list)
                experiment_timeseries[experiment_name] = timeseries_data

    if make_line_plot:
        if per_channel:
            if multi_stat:
                #mutli stat, each channel on their own plot
                plot_experiment_comparison_multi_stat_per_channel(experiment_timeseries, experiment_list, output_directory, stat_pair, array_metrics_list, sensor_name, sat_name, color_list, y_min, y_max)
            else:
                #singular stat, each channel on their own plot
                plot_experiment_comparison_per_channel(experiment_timeseries, experiment_list, output_directory, sensor_name, sat_name, color_list, y_min, y_max)
        else:
            if multi_stat:
                #multi stat, all channels same plot
                plot_experiment_comparison_multi_stat_all_channel(experiment_timeseries, experiment_list, output_directory, stat_pair, array_metrics_list, sensor_name, sat_name, color_list, y_min, y_max) 
            else: 
                #single stat, all channels same plot 
                plot_experiment_comparison(experiment_timeseries, experiment_list, output_directory, channel_list, sensor_name, sat_name, color_list, y_min, y_max) 

    else:
        timeseries_data.print_init_time()

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

#This function plots all channels on the same plot, but each stat and sensor combo gets it's own plot
def plot_experiment_comparison(timeseries_dict, experiment_list, output_dir, channel_list, sensor_name, sat_name, expt_colors=None, y_min=None, y_max=None):
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
                    time_valid = timeseries_obj.timestamp_dict.get(stat_label, {}).get(sensor_label, {})
                    for channel, timestamps in time_valid.items(): 
                        values = timeseries_obj.value_dict.get(stat_label, {}).get(sensor_label, {}).get(channel, [])
                        # Check if data exists for the stat_label and sensor_label
                        if timestamps and values:
                            # Plot the data for this experiment
                            color = expt_colors[i] if expt_colors else None
                            experiment_label = experiment_name
                            if experiment_name in friendly_names_dict:
                                experiment_label = friendly_names_dict[experiment_name]
                            plt.plot(timestamps, values, label=f"{experiment_label} - Ch {channel}", alpha=0.6, color=color) #set plot for line or bar for bar or scatter for scatter
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
                plt.title(f'Comparison of {stat_label} and {sensor_label} for {sensor_name} {sat_name} across Experiments', fontsize=18)
            else:
                plt.title(f'Comparison of {stat_label} and {sensor_label} for {sensor_name} {sat_name} across Experiments for Channels {channel_list}', fontsize=18)
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

#This function plots each channel, stat, sensor combo on it's own plot
def plot_experiment_comparison_per_channel(timeseries_dict, experiment_list, output_dir, sensor_name, sat_name, expt_colors=None, y_min=None, y_max=None):
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
    channel_list = first_timeseries_obj.channel_list


    # Loop through each stat_label in the statlabel_list
    for stat_label in statlabel_list:
        # Loop through each sensor_label in the sensorlabel_list
        for sensor_label in sensorlabel_list:
            for channel_label in channel_list:
                plt.figure(figsize=(16, 12), dpi=300)  # Create a new figure for each stat-sensor-channel combination

                # Loop through each experiment in the experiment list
                for i, experiment_name in enumerate(experiment_list):
                    # Access the corresponding GSIStatsTimeSeries object from the dictionary
                    timeseries_obj = timeseries_dict.get(experiment_name)

                    if timeseries_obj:
                        # Safely access the nested dictionary for time_valid and value
                        timestamps = timeseries_obj.timestamp_dict.get(stat_label, {}).get(sensor_label, {}).get(channel_label, [])
                        values = timeseries_obj.value_dict.get(stat_label, {}).get(sensor_label, {}).get(channel_label, [])
                        # Check if data exists for the stat_label and sensor_label
                        if timestamps and values:
                            time_range = (max(timestamps) - min(timestamps)).total_seconds()
                            new_width = min(max(time_range / 50000, 12), 25)
                            fig = plt.gcf()
                            width, height = fig.get_size_inches()
                            if new_width > width:
                                    fig.set_size_inches(new_width, height)
                            # Plot the data for this experiment
                            color = expt_colors[i] if expt_colors else None
                            experiment_label = experiment_name
                            if experiment_name in friendly_names_dict:
                                experiment_label = friendly_names_dict[experiment_name]
                            plt.plot(timestamps, values, label=f"{experiment_label} - Ch {channel_label}", alpha=0.6, color=color) #set plot for line or bar for bar or scatter for scatter
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
                plt.title(f'Comparison of {stat_label} and {sensor_label} for {sensor_name} {sat_name} across Experiments for Channel {channel_label}', fontsize=18)
                plt.legend(fontsize=20)

                # Rotate x-axis labels for readability
                plt.xticks(rotation=45, fontsize=20)
                plt.yticks(fontsize=20)

                # Save the plot to the specified output directory
                plot_filename = f'{stat_label}_{sensor_label}_{channel_label}_comparison.png'
                plot_filepath = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_filepath)

                # Close the plot after saving
                plt.close()

                print(f"Plot saved: {plot_filepath}")

#This function plots stat and sensor combos on the same plot wihtout regard to separate channels (expects channels averaged and a list of which channels are included for title)
def plot_experiment_comparison_multi_stat(timeseries_dict, experiment_list, output_dir, channel_list, stat_pair, array_metrics_list, sensor_name, sat_name, line_colors=None, y_min=None, y_max=None):
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
                        #TODO: update to handle channels?
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
        plt.title(f'{stat_pair[0]} and {stat_pair[1]} for {sensor_name} {sat_name} and Channel(s) {channel_list}', fontsize=18)
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

#This function plots stat, sensor, and channels all on the same plot 
def plot_experiment_comparison_multi_stat_all_channel(timeseries_dict, experiment_list, output_dir, stat_pair, array_metrics_list, sensor_name, sat_name, line_colors=None, y_min=None, y_max=None):
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
    channel_list = list(first_timeseries_obj)[0].channel_list

    # Set default line colors if none are provided
    # if line_colors is None:
    #     line_colors = plt.cm.get_cmap('tab10', len(experiment_list))

    # Loop through each sensor_label in the sensorlabel_list
    for sensor_label in sensorlabel_list:
        plt.figure(figsize=(16, 12), dpi=300)  # Create a new figure for each sensor combination

        for channel in channel_list:
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
                            time_valid = timeseries_obj.timestamp_dict.get(stat_label, {}).get(sensor_label, {}).get(channel, [])
                            value = timeseries_obj.value_dict.get(stat_label, {}).get(sensor_label, {}).get(channel, [])
                            # Check if data exists for the stat_label and sensor_label
                            if time_valid and value:
                                time_range = (max(time_valid) - min(time_valid)).total_seconds()
                                new_width = min(max(time_range / 50000, 12), 25)
                                fig = plt.gcf()
                                width, height = fig.get_size_inches()
                                if new_width > width:
                                        fig.set_size_inches(new_width, height)
                                # Plot the data for this experiment and stat_label with custom color
                                color = line_colors[i][j] if line_colors else None
                                experiment_label = experiment_name
                                stat_friendly = stat_label
                                if experiment_name in friendly_names_dict:
                                    experiment_label = friendly_names_dict[experiment_name]
                                if stat_label in friendly_names_dict:
                                    stat_friendly = friendly_names_dict[stat_label]
                                plt.plot(time_valid, value, label=f'{experiment_label} - {stat_friendly} - Ch {channel}', color=color, alpha=0.6)
                            else:
                                print(f"No data for {stat_label}, {sensor_label}, {channel} in experiment: {experiment_name}")
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
        plt.title(f'{stat_pair[0]} and {stat_pair[1]} for {sensor_name} {sat_name} and Channel(s) {channel_list}', fontsize=18)
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

#This function plots stat and sensor combos on the same plot but each channel receives it's own plot
def plot_experiment_comparison_multi_stat_per_channel(timeseries_dict, experiment_list, output_dir, stat_pair, array_metrics_list, sensor_name, sat_name, line_colors=None, y_min=None, y_max=None):
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
    channel_list = list(first_timeseries_obj)[0].channel_list

    # Set default line colors if none are provided
    # if line_colors is None:
    #     line_colors = plt.cm.get_cmap('tab10', len(experiment_list))

    for channel in channel_list:
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
                            time_valid = timeseries_obj.timestamp_dict.get(stat_label, {}).get(sensor_label, {}).get(channel, [])
                            value = timeseries_obj.value_dict.get(stat_label, {}).get(sensor_label, {}).get(channel, [])
                            # Check if data exists for the stat_label and sensor_label
                            if time_valid and value:
                                time_range = (max(time_valid) - min(time_valid)).total_seconds()
                                new_width = min(max(time_range / 50000, 12), 25)
                                fig = plt.gcf()
                                width, height = fig.get_size_inches()
                                if new_width > width:
                                        fig.set_size_inches(new_width, height)
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
                                print(f"No data for {stat_label}, {sensor_label}, {channel} in experiment: {experiment_name}")
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
            plt.title(f'{stat_pair[0]} and {stat_pair[1]} for {sensor_name} {sat_name} Channel {channel}', fontsize=18)
            plt.legend(fontsize=20)

            #Rotate x-axis labels for readability
            plt.xticks(rotation=45, fontsize=20)
            plt.yticks(fontsize=20)

            # Save the plot to the specified output directory
            plot_filename = f'{sensor_label}_ch{channel}_comparison_{stat_pair[0]}_{stat_pair[1]}.png'
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath)

            # Close the plot after saving
            plt.close()

            print(f"Plot saved: {plot_filepath}")


def main():
    #run()
    run_line_plot()

if __name__=='__main__':
    main()
