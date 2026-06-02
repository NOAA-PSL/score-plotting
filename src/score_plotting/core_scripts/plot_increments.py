"""Copyright 2023 NOAA
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

import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from score_db.expt_metrics import ExptMetricRequest
from score_plotting.attrs.increments_plot_attrs import plot_attrs
from score_plotting.core_scripts.plot_innov_stats import PlotInnovStatsRequest

HOURS_PER_DAY = 24. # hours
UFS_REPLAY_BUCKET = 'noaa-ufs-gefsv13replay-pds'

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Add DA cycle as an argument (optional, default to 6.0)
    parser.add_argument('--da_cycle', type=float, default=6.,
                        help='The DA cycle duration in hours (default: 6.0)')
    
    # Add days to smooth as an argument (optional, default to 8.0)
    parser.add_argument('--days_to_smooth', type=float, default=8.,
                        help='Number of days to smooth (default: 8.0)')
    
    # Make figure_output_path optional (defaults to $HOME)
    parser.add_argument('figure_output_path', type=str, nargs='?',
                        default=pathlib.Path.home(),
                        help='Path to where figures will be saved')
                        
    parser.add_argument(
        '--dark_theme', 
        action='store_true',  # If this argument is provided, dark_theme will be True
        help="Enable dark theme (default is False)"
    )
    
    args = parser.parse_args()

    return args

RequestData = namedtuple('RequestData', ['datetime_str', 'experiment',
                                         'metric_format_str', 'metric',
                                         'stat',
                                         'time_valid'],)

plot_control_dict1 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '1999-01-01 00:00:00',
                                    'start': '1994-01-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'replay_stream1',
                                      'wallclock_start': '2023-07-08 16:25:57'}],
                     'fig_base_fn': 'increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc',
                                                  'u_inc_ocn','v_inc_ocn',
                                                  'u_inc_atm','v_inc_atm',
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict2 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2005-01-01 00:00:00',
                                    'start': '1999-01-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'replay_stream2',
                                      'wallclock_start': '2023-07-24 17:56:40'}],
                     'fig_base_fn': 'increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc', 
                                                  'u_inc_ocn','v_inc_ocn',
                                                  'u_inc_atm','v_inc_atm',
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict3 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2010-01-01 00:00:00',
                                    'start': '2005-01-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'replay_stream3',
                                      'wallclock_start': '2023-01-22 09:22:05'}],
                     'fig_base_fn': 'increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc','u_inc_ocn',
                                                  'v_inc_ocn', 'u_inc_atm','v_inc_atm', 
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict4 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2015-01-01 00:00:00',
                                    'start': '2010-01-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'replay_stream4',
                                      'wallclock_start': '2023-01-22 09:22:05'}],
                     'fig_base_fn': 'increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc','u_inc_ocn',
                                                  'v_inc_ocn', 'u_inc_atm','v_inc_atm', 
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict5 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2020-01-01 00:00:00',
                                    'start': '2015-01-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'replay_stream5',
                                      'wallclock_start': '2023-07-08 06:20:22'}],
                     'fig_base_fn': 'increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc','u_inc_ocn',
                                                  'v_inc_ocn', 'u_inc_atm','v_inc_atm',
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
plot_control_dict6 = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'end': '2024-01-01 00:00:00',
                                    'start': '2020-01-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'replay_stream6',
                                      'wallclock_start': '2023-07-24 20:29:23'}],
                     'fig_base_fn': 'increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc','u_inc_ocn',
                                                  'v_inc_ocn', 'u_inc_atm','v_inc_atm',
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}
                     
plot_control_dict_forward_ext = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'start': '2024-10-01 00:00:00',
                                    'end': '2026-04-01 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'ufs_replay_ext',
                                      'wallclock_start': '2024-10-01 00:00:00'}],
                     'fig_base_fn': 'forward_ext_increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc','u_inc_ocn',
                                                  'v_inc_ocn', 'u_inc_atm','v_inc_atm',
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}

plot_control_dict_ext = {'date_range': {'datetime_str': '%Y-%m-%d %H:%M:%S',
                                    'start': '1978-10-01 00:00:00',
                                    'end': '2025-09-30 00:00:00'},
                     'db_request_name': 'expt_metrics',
                     'method': 'GET',
                     'experiments': [{'graph_color': 'black',
                                      'graph_label': 'increment',
                                      'name': 'ufs_replay_ext',
                                      'wallclock_start': '2024-10-01 00:00:00'}],
                     'fig_base_fn': 'increment',
                     'stat_groups': [{'cycles': [0, 21600, 43200, 64800],
                                      'stats': ['mean', 'RMS'],
                                      'metrics': ['pt_inc', 's_inc','u_inc_ocn',
                                                  'v_inc_ocn', 'u_inc_atm','v_inc_atm',
                                                  'SSH', 'Salinity', 'Temperature',
                                                  'Speed of Currents', 'o3mr_inc',
                                                  'sphum_inc', 'T_inc', 'delp_inc',
                                                  'delz_inc'],
                                      'stat_group_frmt_str':
                                      'metric_type_{stat}_{metric}'}],
                     'work_dir': parse_arguments().figure_output_path}

def get_experiment_increments(request_data):
    
    expt_metric_name = request_data.metric_format_str.replace(
                                                        '{metric}', 
                                                        request_data.metric)

    if (request_data.metric[-4:] == "_ocn" or request_data.metric[-4:] == "_atm"):
        metric_measurement_type= request_data.metric[:-4]
        metric_measurement_name= request_data.stat+"_"+request_data.metric
    else:
        metric_measurement_type= request_data.metric
        metric_measurement_name= request_data.stat+"_"+request_data.metric

    expt_metric_name = expt_metric_name.replace(
        '{stat}', request_data.stat
    )
   
    time_valid_from = datetime.strftime(request_data.time_valid.start, 
                                        request_data.datetime_str)

    time_valid_to = datetime.strftime(request_data.time_valid.end, 
                                      request_data.datetime_str)
    request_dict = {'name': 'expt_metrics', 'method': 'GET',
                    'params': {'datestr_format': '%Y-%m-%d %H:%M:%S',
                               'filters':
                                 {'experiment':
                                   {'name': {
                                      'exact': request_data.experiment['name']['exact']},
                                    'wallclock_start':
                                      {'from': request_data.experiment['wallclock_start']['from'],
                                       'to': request_data.experiment['wallclock_start']['to']}},
                                  'metric_types': {'name': {'exact': [metric_measurement_name]},
                                                   'measurement_type': {'exact': [metric_measurement_type]},
                                                   'stat_type': {'exact': [request_data.stat]}},
                                  'regions': {'rgs_name': {'exact': ['global']}},
                                  'time_valid': {'from': time_valid_from,
                                                 'to': time_valid_to}},
                                 'ordering': [{'name': 'time_valid', 'order_by': 'asc'}]}}

    print(f'request_dict: {request_dict}')

    emr = ExptMetricRequest(request_dict)
    result = emr.submit()
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

def build_fig_dest(work_dir, fig_base_fn, stat, metric, date_range,
                   experiment_name=None, append_date_range=False):
    
    start = datetime.strftime(date_range.start, '%Y%m%dT%HZ')
    end = datetime.strftime(date_range.end, '%Y%m%dT%HZ')
    dest_fn = fig_base_fn
    
    if append_date_range:
        dest_fn += f'_{stat}_{metric}_{start}_to_{end}.png'
    else:
        dest_fn += f'_{stat}_{metric}.png'
    
    if experiment_name is not None:
        dest_full_path = os.path.join(work_dir, experiment_name, dest_fn)
    else:
        dest_full_path = os.path.join(work_dir, dest_fn)
    
    parent_dir = pathlib.Path(dest_full_path).parent
    pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)
    
    return dest_full_path

def save_figure(dest_full_path):
    print(f'saving figure to {dest_full_path}')
    plt.savefig(dest_full_path, dpi=300)

def plot_increments(experiments, stat, metric, metrics_df, work_dir, fig_base_fn,
                     date_range, append_date_range=False):
    args = parse_arguments()
    
    if args.dark_theme:
        default_plot_color = 'white'#'#CFB87C'
        fill_color = '#565A5C'
    else:
        default_plot_color = experiments[0]['graph_color']
        fill_color = '#A2A4A3'
    
    time_domain = pd.Series(
        data = np.nan,
        index = pd.date_range(
            start = date_range.start,
            end = date_range.end,
            freq = pd.Timedelta(hours=args.da_cycle)
        )
    )
    window_size = pd.Timedelta(hours=HOURS_PER_DAY * args.days_to_smooth)

    if not isinstance(metrics_df, DataFrame):
        msg = 'Input data to plot_increments must be type pandas.DataFrame '\
            f'was actually type: {type(metrics_df)}'
        raise TypeError(msg)

    plt_attr_key = 'increment'
    pa = plot_attrs[plt_attr_key]
    (fig, ax) = build_base_figure()
    
    if args.dark_theme:
        ax.axhline(color='#A2A4A3', lw=0.75)
    else:
        ax.axhline(color='black', lw=0.75)

    metrics_to_show = metrics_df.drop_duplicates(subset='time_valid', keep='last')
    expt_name = experiments[0]['name']['exact']

    timestamps = list()
    labels = list()
    values = list()
    colors = list()
    cycle_labels = list()

    for row in metrics_to_show.itertuples():
        if row.time_valid >= date_range.start and row.time_valid < date_range.end:
            values.append(row.value)
            timestamps.append(row.time_valid)
            '''
            labels.append('%02d-%02d-%04d' % (row.time_valid.month,
                                              row.time_valid.day,
                                              row.time_valid.year,
                                          ))
            '''
            cycle_labels.append('%dZ' % row.time_valid.hour)
            
            if row.time_valid.hour == 0:
                colors.append('lightcoral')
            elif row.time_valid.hour == 6:
                colors.append('yellowgreen')
            elif row.time_valid.hour == 12:
                colors.append('skyblue')
            elif row.time_valid.hour == 18:
                colors.append('orchid')
            else:
                colors.append(default_plot_color)

    myLabel = set(cycle_labels)

    if args.dark_theme:
        plt.grid()
    
    values_timeseries = pd.Series(
        data = values,
        index = timestamps
    ).combine_first(time_domain)    
        
    plt.fill_between(
        values_timeseries.index,
        np.nan_to_num(values_timeseries.values),
        interpolate=True,
        step='mid',
        edgecolor='none',
        lw=0,
        color=fill_color,
        zorder=2
    )
    
    for i in range(len(myLabel)):
        """ Plot the first unique cycles to format the legend
        """
        plt.scatter(timestamps[i], values[i], #s=1,
                c=colors[i], marker='|',
                alpha=0.9, label=cycle_labels[i],
                linewidths=0.5,
                zorder=3)
    
    # proceed with onward
    plt.scatter(timestamps[len(myLabel):], values[len(myLabel):], #s=1,
                c=colors[len(myLabel):], marker='|', alpha=0.9,
                linewidths=0.5,
                zorder=3)
    
    values_smooth = values_timeseries.rolling(
                        window=window_size,
                        min_periods=int(np.around(args.days_to_smooth)),
                        center=True).mean()
    
    plt.plot(values_smooth.index,
             values_smooth.values,
             ls='-',
             marker='none',
             color=default_plot_color,
             alpha=0.9,
             lw=1.5,
             label=f'{int(args.days_to_smooth)} day SMA',
             zorder=4)
    
    format_figure(ax, pa)
    '''
    ymin, ymax = ax.get_ylim()
    if stat == 'RMS' and ymin < 0:
        ax.set_ylim(0, None)
    if stat == 'RMS' and np.nanmin(values) > 0:
        ax.set_ylim(np.nanmin(values, None))
    if "_inc" not in metric and np.nanmin(values) > 0:
        ax.set_ylim(np.nanmin(values, None))
    '''
    ax.set_ylim(np.nanmin(values), np.nanmax(values))

    if "_inc" not in metric:
        # this is not an increment
        if row.metric_unit == 'C':
            if metric=='pt':
                plot_basic_title = 'ocean potential temperature'
                expt_graph_label = 'Potential temperature ($^\circ$C)'
            elif metric=='Temperature':
                plot_basic_title = 'ocean temperature'
                expt_graph_label = 'Temperature ($^\circ$C)'
        elif row.metric_unit == 'm':
            if metric =='SSH':
                plot_basic_title = 'sea surface height (SSH)'
                expt_graph_label = 'SSH (m)'
        elif row.metric_unit == 'PSU':
            plot_basic_title = 'ocean salinity'
            expt_graph_label = 'Salinity (psu)'    
        elif row.metric_unit == 'm/s':
            if metric=='Speed of Currents':
                plot_basic_title = 'speed of currents'
                expt_graph_label = 'Speed (m s$^{-1}$)'
    else:
        if row.metric_unit == 'C':
            if metric=='pt_inc':
                plot_basic_title = 'ocean potential temperature increment'
                expt_graph_label = 'Potential temperature increment ($^\circ$C)'
            elif metric=='T_inc':
                plot_basic_title = 'air temperature increment'
                expt_graph_label = 'Temperature increment ($^\circ$C)'
        elif row.metric_unit == 'PSU':
            plot_basic_title = 'ocean salinity increment'
            expt_graph_label = 'Salinity increment (psu)'
        elif row.metric_unit == 'm/s':
            expt_graph_label = 'Speed increment (m s$^{-1}$)'
            if metric=='u_inc_atm':
                plot_basic_title = 'westerly wind speed increment'
            elif metric=='v_inc_atm':
                plot_basic_title = 'southerly wind speed increment'
            elif metric=='u_inc_ocn':
                plot_basic_title = 'horizontal x-direction ocean speed increment'
            elif metric=='v_inc_ocn':
                plot_basic_title = 'horizontal y-direction ocean speed increment'
        elif row.metric_unit == 'kg/kg':
            if metric=='o3mr_inc':
                plot_basic_title = 'ozone mixing ratio increment'
                expt_graph_label = 'Mixing ratio increment (kg kg$^{-1}$)'
            elif metric=='sphum_inc':
                plot_basic_title = 'specific humidity increment'
                expt_graph_label = 'Specific humidity increment (kg kg$^{-1}$)'
        if metric=='delp_inc':
            plot_basic_title = 'pressure differential increment'
            expt_graph_label = 'Pressure differential increment (Pa)'
        elif row.metric_unit == 'm':
            if metric=='delz_inc':
                plot_basic_title = 'geometric layer height differential increment'
                expt_graph_label = 'Layer height differential increment (m)'
    
    if expt_name == 'ufs_replay_ext':
        plot_expt_title = UFS_REPLAY_BUCKET
    else:
        plot_expt_title = expt_name 
    
    plt.title(f'Global {stat} {plot_basic_title}', loc='left')
    today = date.today()
    fig.suptitle(plot_expt_title, x=0.02, ha='left')
    plt.title(today, loc='right')
    plt.ylabel(expt_graph_label)
    
    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    
    if pd.Timedelta(date_range.end - date_range.start) > pd.Timedelta(days=10957):
        # date range is greater than 30 years; set minor tick interval to 3 months
        month_interval = 3
    else:
        # date range is less than or equal to 30 years; set minor tick interval to 1 month
        month_interval = 1
    
    month_locator = mdates.MonthLocator(interval=month_interval)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(month_locator)
    
    fig_fn = build_fig_dest(work_dir, fig_base_fn, stat, metric, date_range,
                            experiment_name=expt_name, append_date_range=append_date_range)

    plt.tight_layout()
    save_figure(fig_fn)
    plt.close()

@dataclass
class PlotIncrementRequest(PlotInnovStatsRequest):
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
                for stat in stat_group.stats:
                    m_df = DataFrame()
                    for experiment in self.experiments:
                        request_data = RequestData(
                            self.datetime_str,
                            experiment,
                            stat_group.stat_group_frmt_str,
                            metric, stat,
                            self.date_range)
                        
                        try:
                            e_df = get_experiment_increments(request_data)
                            e_df = e_df.sort_values(['time_valid', 'created_at'])
                            m_df = pd.concat([m_df, e_df], axis=0)
                            plot_yes = True
                        except KeyError:
                            print('no records found for %s %s, skipping' % (stat, metric))
                            plot_yes = False
                    if plot_yes:
                        plot_increments(
                            self.experiments,
                            stat,
                            metric,
                            m_df,
                            self.work_dir,
                            self.fig_base_fn,
                            self.date_range,
                            append_date_range=False)

if __name__=='__main__':
    args = parse_arguments()
    if args.dark_theme:
        style_file = 'dark_theme.mplstyle'
    else:
        style_file = 'half_horizontal.mplstyle'
    
    style_file_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(),
                                   'style_lib', style_file)
    plt.style.use(style_file_path)
    
    for i, plot_control_dict in enumerate([#plot_control_dict1,
                                           #plot_control_dict2,
                                           #plot_control_dict3,
                                           #plot_control_dict4,
                                           #plot_control_dict5,
                                           #plot_control_dict6
                                           plot_control_dict_ext,
                                           plot_control_dict_forward_ext]):
        plot_request = PlotIncrementRequest(plot_control_dict)
        plot_request.submit()
