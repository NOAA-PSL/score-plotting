from datetime import datetime

from score_db import score_db_base

def extract_unique_stats(strings):
    # Create sets to store unique values in the second and last positions
    second_position_set = set()
    last_position_set = set()
    
    # Iterate over the set of strings
    for s in strings:
        parts = s.split('_')
        
        # Add the second and last elements to their respective sets
        if len(parts) > 1: #and parts[-1] != 'None':  # Ensure there are at least 2 parts, 
            second_position_set.add('_'.join(parts[1:-2]))
            last_position_set.add(parts[-1])
    
    # Convert sets to sorted lists to maintain a consistent order
    second_position_list = sorted(list(second_position_set))
    last_position_list = sorted(list(last_position_set))
    
    # Combine the two lists into a 2D array (list of lists)
    unique_positions = [second_position_list, last_position_list]
    
    return unique_positions

def get_data_frame(experiment_list,
                   array_metric_list,
                   start_date='1979-01-01 00:00:00',
                   stop_date='2026-01-01 00:00:00',
                   select_sat_name=False,
                   sat_name=None):
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
                                 experiment_list}
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

    request_dict['params']['filters']['array_metric_types'] = {
        'name': {'exact': array_metric_list}
    }

    if select_sat_name:
        request_dict['params']['filters']['sat_meta'] = {
            'sat_name': {'exact': sat_name}
        }

    db_action_response = score_db_base.handle_request(request_dict)    
    data_frame = db_action_response.details['records']
    
    # sort by timestamp, created at
    data_frame.sort_values(by=['expt_name',
                               'metric_name',
                               'sat_short_name',
                               'time_valid',
                               'created_at'], 
                               inplace=True)

    # remove duplicate data
    data_frame.drop_duplicates(subset=['expt_name',
                                       'metric_name',
                                       'sat_short_name',
                                       'time_valid'], 
                               keep='last', inplace=True)
                               
    return data_frame

class GSIStatsTimeSeries(object):
    def __init__(self,
                 start_date,
                 stop_date,
                 data_frame=None,
                 input_data_frame=False,
                 experiment_name = 'scout_run_v1',#
                                   #'scout_runs_gsi3dvar_1979stream',#
                select_array_metric_types = True,
                array_metric_types='%',
                select_sat_name = False,
                sat_name = None,
                experiment_id=None,
                ):
        """Download metrics data for given experiment name
        """
        self.init_datetime = datetime.now()
        self.experiment_name = experiment_name
        self.select_array_metric_types = select_array_metric_types
        self.array_metric_types = array_metric_types
        self.select_sat_name = select_sat_name
        self.sat_name = sat_name
        self.experiment_id = experiment_id
        if input_data_frame and type(self.array_metric_types) == str:
            self.data_frame = data_frame[(
                data_frame['expt_name'] == self.experiment_name) &            
                (data_frame['metric_name'] == self.array_metric_types)]
        
        else:
            self.data_frame = get_data_frame(
                [self.experiment_name],
                [self.array_metric_types],
                start_date=start_date,
                stop_date=stop_date,
                select_sat_name=self.select_sat_name,
                sat_name=self.sat_name)
        
    def build(self, all_channel_max=False, all_channel_mean=False, by_channel=True):
        self.unique_stat_list = extract_unique_stats(
                                            set(self.data_frame['metric_name']))
        
        self.timestamp_dict = dict()
        #self.timelabel_dict = dict()
        self.value_dict = dict()
        for i, stat_name in enumerate(self.unique_stat_list[0]):
            for j, gsi_stage in enumerate(self.unique_stat_list[1]):
                self.timestamp_dict[f'{stat_name}_GSIstage_{gsi_stage}'] = dict()
                #self.timelabel_dict[f'{stat_name}_GSIstage_{gsi_stage}'] = dict()
                self.value_dict[f'{stat_name}_GSIstage_{gsi_stage}'] = dict()
                
        for key in self.value_dict.keys():
            for sat_short_name in set(self.data_frame.sat_short_name):
                for instrument_name in set(
                                    self.data_frame.metric_instrument_name):
                    sensor_label = f'{sat_short_name}_{instrument_name}'
                    
                    self.timestamp_dict[key][sensor_label] = list()
                    #self.timelabel_dict[key][sensor_label] = list()
                    self.value_dict[key][sensor_label] = list()
        
        self.sensorlabel_dict = dict()
        yval = 0
        for row in self.data_frame.itertuples():
            metric_name_parts = row.metric_name.split('_')

            if metric_name_parts[0] == row.metric_instrument_name and row.expt_name == self.experiment_name:
                stat_name = '_'.join(metric_name_parts[1:-2])
                gsi_stage = metric_name_parts[-1]
                
                stat_label = f'{stat_name}_GSIstage_{gsi_stage}'
                
                sensor_label = f'{row.sat_short_name}_{row.metric_instrument_name}'
                timestamp = row.time_valid#.timestamp()
                
                if False:
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
                #self.timelabel_dict[stat_label][sensor_label].append(time_label)
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


    def flatten_by_channel(self, channel_list):
        self.unique_stat_list = extract_unique_stats(
                                            set(self.data_frame['metric_name']))
        
        self.timestamp_dict = dict()
        self.timelabel_dict = dict()
        self.value_dict = dict()
        
        self.sensorlabel_list = list()
        self.statlabel_list = list()
        self.channel_list = list()
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
                
                for i, channel in enumerate(row.array_index_values):
                    if channel_list is not None and channel not in channel_list: 
                        continue 

                    value = np.nan if row.value[i] is None else row.value[i]

                    # Check if stat_label exists in timestamp_dict
                    if stat_label not in self.timestamp_dict:
                        self.timestamp_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                    # Check if sensor_label exists under stat_label in timestamp_dict
                    if sensor_label not in self.timestamp_dict[stat_label]:
                        self.timestamp_dict[stat_label][sensor_label] = {}  # Second level dictionary

                    #Check if channel 
                    if channel not in self.timestamp_dict[stat_label][sensor_label]:
                        self.timestamp_dict[stat_label][sensor_label][channel] = [] #Empty list for channel level values

                    # Check if stat_label exists in timelabel_dict
                    if stat_label not in self.timelabel_dict:
                        self.timelabel_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                    # Check if sensor_label exists under stat_label in timelabel_dict
                    if sensor_label not in self.timelabel_dict[stat_label]:
                        self.timelabel_dict[stat_label][sensor_label] = {}  # Second level dictionary

                    if channel not in self.timelabel_dict[stat_label][sensor_label]:
                        self.timelabel_dict[stat_label][sensor_label][channel] = [] #Empty list for channel level values
                    
                    # Check if stat_label exists in timelabel_dict
                    if stat_label not in self.value_dict:
                        self.value_dict[stat_label] = {}  # Create the first level dictionary for stat_label

                    # Check if sensor_label exists under stat_label in timelabel_dict
                    if sensor_label not in self.value_dict[stat_label]:
                        self.value_dict[stat_label][sensor_label] = {}  # Second level dictionary

                    if channel not in self.value_dict[stat_label][sensor_label]:
                        self.value_dict[stat_label][sensor_label][channel] = []  # Empty list for channel level values  

                    #print(gsi_stage, stat_name, sensor_label, time_label, value)
                    self.timestamp_dict[stat_label][sensor_label][channel].append(timestamp)
                    self.timelabel_dict[stat_label][sensor_label][channel].append(time_label)
                    self.value_dict[stat_label][sensor_label][channel].append(value)
                    
                    if not sensor_label in self.sensorlabel_list:
                        self.sensorlabel_list.append(sensor_label)
                    
                    if not stat_label in self.statlabel_list:
                        self.statlabel_list.append(stat_label)

                    if not channel in self.channel_list:
                        self.channel_list.append(channel)
        
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

    def plot_line_plot_by_channel(self, stat_label, sensor_label, experiment_name, channels_to_plot, y_min=None, y_max=None):
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
                channel_values = self.value_dict[stat_label][sensor_label][channel]

                # Extract the corresponding timestamps
                timestamps = self.timestamp_dict[stat_label][sensor_label][channel]

                # Ensure the channel is valid and plot the values
                if channel_values is not None:
                    plt.plot(timestamps, channel_values, label=f'Channel {channel}', alpha=0.7)
                else:
                    print(f"Channel {channel} not found for {stat_label}, {sensor_label}")
                    
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