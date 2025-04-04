#!/usr/bin/env python

"""
Copyright 2023 NOAA
All rights reserved.

Collection of methods to facilitate handling of score db requests
"""

import os
import pathlib
import argparse
from dataclasses import dataclass, field
from collections import namedtuple
from datetime import datetime
from datetime import date

import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from score_db.expt_file_counts import ExptFileCountRequest
from score_plotting.attrs.file_counts_plot_attrs import plot_attrs
from score_plotting.core_scripts.plot_innov_stats import PlotInnovStatsRequest

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Make figure_output_path optional (defaults to $HOME)
    parser.add_argument('figure_output_path', type=str, nargs='?',
                        default=pathlib.Path.home(),
                        help='Path to where figures will be saved')
                        
    # Add DA cycle as an argument (optional, default to 6.0)
    parser.add_argument('--da_cycle', type=float, default=6.,
                        help='The DA cycle duration in hours (default: 6.0)')
    
    parser.add_argument(
        '--dark_theme', 
        action='store_true',  # If this argument is provided, dark_theme will be True
        help="Enable dark theme (default is False)"
    )
    
    args = parser.parse_args()

    return args

RequestData = namedtuple('RequestData', ['datetime_str', 'experiment',
                                         'metric_format_str', 'metric',
                                         'time_valid'],)

plot_control_dict1 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '1981-07-01 00:00:00',
                                    'start': '1979-01-01 00:00:00'},
                     'db_request_name': 'expt_file_counts',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'Number of files',
                                      'name': 'scout_runs_gsi3dvar_1979stream',
                                      'wallclock_start': '2023-12-15 17:30:10'}],
                     'fig_base_fn': 'files',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'metrics': ['count'],
                                      'stat_group_frmt_str':
                                      'file_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}

plot_control_dict2 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2000-01-01 00:00:00',
                                    'start': '1994-01-01 00:00:00'},
                     'db_request_name': 'expt_file_counts',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'Number of files',
                                      'name': 'scout_runs_gsi3dvar_rod',
                                      'wallclock_start': '2024-01-14 21:21:35'}],
                     'fig_base_fn': 'files',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'metrics': ['count'],
                                      'stat_group_frmt_str':
                                      'file_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}

plot_control_dict3 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2010-01-01 00:00:00',
                                    'start': '2005-01-01 00:00:00'},
                     'db_request_name': 'expt_file_counts',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'Number of files',
                                      'name': 'replay_stream3',
                                      'wallclock_start': '2023-01-22 09:22:05'}],
                     'fig_base_fn': 'files',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'metrics': ['count'],
                                      'stat_group_frmt_str':
                                      'file_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict4 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2015-01-01 00:00:00',
                                    'start': '2010-01-01 00:00:00'},
                     'db_request_name': 'expt_file_counts',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'Number of files',
                                      'name': 'replay_stream4',
                                      'wallclock_start': '2023-01-22 09:22:05'}],
                     'fig_base_fn': 'files',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'metrics': ['count'],
                                      'stat_group_frmt_str':
                                      'file_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict5 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2020-01-01 00:00:00',
                                    'start': '2015-01-01 00:00:00'},
                     'db_request_name': 'expt_file_counts',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'Number of files',
                                      'name': 'replay_stream5',
                                      'wallclock_start': '2023-07-08 06:20:22'}],
                     'fig_base_fn': 'files',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'metrics': ['count'],
                                      'stat_group_frmt_str':
                                      'file_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict6 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2024-01-01 00:00:00',
                                    'start': '2020-01-01 00:00:00'},
                     'db_request_name': 'expt_file_counts',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'Number of files',
                                      'name': 'replay_stream6',
                                      'wallclock_start': '2023-07-24 20:29:23'}],
                     'fig_base_fn': 'files',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'metrics': ['count'],
                                      'stat_group_frmt_str':
                                      'file_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
                     
plot_control_dict_ext = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2025-09-30 00:00:00',
                                    'start': '2020-10-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increments',
                                      'name': 'ufs_replay_ext',
                                      'wallclock_start': '2024-10-01 00:00:00'}],
                     'fig_base_fn': 'files',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'metrics': ['count'],
                                      'stat_group_frmt_str':
                                      'file_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}

def get_experiment_file_counts(request_data):
    
    expt_metric_name = request_data.metric_format_str.replace(
                                                        '{metric}', 
                                                        request_data.metric)

    time_valid_from = datetime.strftime(request_data.time_valid.start, 
                                        request_data.datetime_str)

    time_valid_to = datetime.strftime(request_data.time_valid.end, 
                                      request_data.datetime_str)
    
    request_dict = {'name': 'expt_file_counts',
                    'method': 'GET',
                    'params': {
                        'filters': {
                            'experiment': {
                                'experiment_name': {
                                    'exact':
                                        request_data.experiment['name']['exact']
                                }
                            }  
                        }
                    }
    }
                                    
    print(f'request_dict: {request_dict}')

    efcr = ExptFileCountRequest(request_dict)
    result = efcr.submit()

    return result.details['records']

def build_base_figure():
    
    fig, ax = plt.subplots()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='x', which='both', bottom=True, top=False,
                    labelbottom=True)
    
    return(fig, ax)

def format_figure(ax, pa):
    ax.set_xlim([pd.Timestamp(plot_control_dict['date_range']['start']),
                 pd.Timestamp(plot_control_dict['date_range']['end'])])
    ax.set_ylim([pa.axes_attrs.ymin, pa.axes_attrs.ymax])
    plt.xlabel(xlabel=pa.xlabel.label,
               horizontalalignment=pa.xlabel.horizontalalignment)
    
    plt.ylabel(ylabel=pa.ylabel.label,
               horizontalalignment=pa.ylabel.horizontalalignment)

    plt.legend(loc=pa.legend.loc,
               fancybox=pa.legend.fancybox,
               edgecolor=pa.legend.edgecolor,
               framealpha=pa.legend.framealpha,
               shadow=pa.legend.shadow,
               facecolor=pa.legend.facecolor)

def build_fig_dest(work_dir, fig_base_fn, metric, date_range,
                   experiment_name=None, append_date_range=False):
    
    start = datetime.strftime(date_range.start, '%Y%m%dT%HZ')
    end = datetime.strftime(date_range.end, '%Y%m%dT%HZ')
    dest_fn = fig_base_fn
    
    if append_date_range:
        dest_fn += f'_{metric}_{start}_to_{end}.png'
    else:
        dest_fn += f'_{metric}.png'
    
    if experiment_name is not None:
        dest_full_path = os.path.join(work_dir, experiment_name, dest_fn)
    else:
        dest_full_path = os.path.join(work_dir, dest_fn)
    
    parent_dir = pathlib.Path(dest_full_path).parent
    pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)
    
    return dest_full_path

def save_figure(dest_full_path):
    print(f'saving figure to {dest_full_path}')
    plt.savefig(dest_full_path, dpi=600)

def plot_file_counts(experiments, metric, metrics_df, work_dir, fig_base_fn,
                     date_range):
    args = parse_arguments()
    
    if args.dark_theme:
        default_plot_color = 'white'#'#CFB87C'
        fill_color = '#565A5C'
    else:
        default_plot_color = 'black'
        fill_color = '#A2A4A3'
    
    da_cycle = args.da_cycle
    
    file_count = open(os.path.join(work_dir,'File_count_unexpected.txt'),'a')
    time_domain = pd.Series(
        data = np.nan,
        index = pd.date_range(
            start = date_range.start,
            end = date_range.end,
            freq = pd.Timedelta(hours=da_cycle)
        )
    )
    if not isinstance(metrics_df, DataFrame):
        msg = 'Input data to plot_file_counts must be type pandas.DataFrame '\
            f'was actually type: {type(metrics_df)}'
        raise TypeError(msg)
    
    plt_attr_key = f'{metric}'
    pa = plot_attrs[plt_attr_key]

    metrics_to_show = metrics_df.drop_duplicates(subset='cycle', keep='last')
    
    (fig, ax) = build_base_figure()

    expt_name = experiments[0]['name']['exact']
    
    timestamps = list()
    labels = list()
    counts = list()
    colors = list()
    cycle_labels = list()
    
    for row in metrics_to_show.itertuples():
        if row.cycle >= date_range.start and row.cycle < date_range.end:
            counts.append(row.count)
            timestamps.append(row.cycle)
            '''
            labels.append('%02d-%02d-%04d' % (row.cycle.month,
                                              row.cycle.day,
                                              row.cycle.year,
                                              ))
            '''
            cycle_labels.append('%dZ' % row.cycle.hour)
            if row.cycle.hour == 0:
                colors.append('lightcoral')
                if(row.count < 31):
                    file_count.write('%s count %02d/31 %02d-%02d-%04d %02d\n' % (expt_name,row.count,
					      row.cycle.month,
                                              row.cycle.day,
                                              row.cycle.year,
                                              row.cycle.hour
                                          ))
            elif row.cycle.hour == 6:
                colors.append('yellowgreen')
                if(row.count < 74):
                    file_count.write('%s count %02d/74 %02d-%02d-%04d %02d\n' % (expt_name,row.count,
                                              row.cycle.month,
                                              row.cycle.day,
                                              row.cycle.year,
                                              row.cycle.hour
                                          ))
            elif row.cycle.hour == 12:
                colors.append('skyblue')
                if(row.count < 31):
                    file_count.write('%s count %02d/31 %02d-%02d-%04d %02d\n' % (expt_name,row.count,
                                              row.cycle.month,
                                              row.cycle.day,
                                              row.cycle.year,
                                              row.cycle.hour
                                          ))
            elif row.cycle.hour == 18:
                colors.append('orchid')
                if(row.count < 29):
                    file_count.write('%s count %02d/29 %02d-%02d-%04d %02d\n' % (expt_name,row.count,
                                              row.cycle.month,
                                              row.cycle.day,
                                              row.cycle.year,
                                              row.cycle.hour
                                          ))
                                          
            else:
                colors.append(default_plot_color)
    
    if args.dark_theme:
        plt.grid()
        
    counts_timeseries = pd.Series(
        data = counts,
        index = timestamps
    ).combine_first(time_domain)

    plt.fill_between(
        counts_timeseries.index,
        np.nan_to_num(counts_timeseries.values),
        interpolate=True,
        step='mid',
        edgecolor='none',
        lw=0,
        color=fill_color,
        zorder=2
    )
    
    myLabel = set(cycle_labels) 
    for i in range(len(myLabel)):
        """ Plot the first unique cycles to format the legend
        """
        plt.scatter(timestamps[i], counts[i], #s=1,
                c=colors[i], marker='|',
                alpha=0.9, label=cycle_labels[i],
                linewidths=0.5,
                zorder=3)
    
    # proceed with onward
    plt.scatter(timestamps[len(myLabel):], counts[len(myLabel):], #s=1,
                c=colors[len(myLabel):], marker='|', alpha=0.9,
                linewidths=0.5,
                zorder=3)
    
    format_figure(ax, pa)

    today = date.today()
    plt.title(today, loc = "right")
    
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    month_locator = mdates.MonthLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(month_locator)
    plt.title(expt_name)
    format_figure(ax, pa)
    fig_fn = build_fig_dest(work_dir, fig_base_fn, metric, date_range,
                            experiment_name=expt_name)  

    save_figure(fig_fn)
    plt.close()

@dataclass
class PlotFileCountRequest(PlotInnovStatsRequest):
    def submit(self):
        master_list = []
        n_hours = 6
        n_days = 0

        finished = False
        loop_count = 0
        for stat_group in self.stat_groups:
            metrics_data = []
            # gather experiment metrics data for experiment and date range
            for metric in stat_group.metrics:
                m_df = DataFrame()
                for experiment in self.experiments:
                    request_data = RequestData(
                        self.datetime_str,
                        experiment,
                        stat_group.stat_group_frmt_str,
                        metric,
                        self.date_range)
                        
                    e_df = get_experiment_file_counts(request_data)
                    e_df = e_df.sort_values(['cycle', 'created_at'])
                    m_df = pd.concat([m_df, e_df], axis=0)

                plot_file_counts(
                    self.experiments,
                    metric,
                    m_df,
                    self.work_dir,
                    self.fig_base_fn,
                    self.date_range)

if __name__=='__main__':
    args = parse_arguments()
    if args.dark_theme:
        style_file = 'dark_theme.mplstyle'
    else:
        style_file = 'half_horizontal.mplstyle'
    
    style_file_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(),
                                   'style_lib', style_file)
    plt.style.use(style_file_path)
    for i, plot_control_dict in enumerate([plot_control_dict1,
                                           plot_control_dict2,
                                           plot_control_dict3,
                                           plot_control_dict4,
                                           plot_control_dict5,
                                           plot_control_dict6,
                                           plot_control_dict_ext
                                         ]):
        plot_request = PlotFileCountRequest(plot_control_dict)
        plot_request.submit()
