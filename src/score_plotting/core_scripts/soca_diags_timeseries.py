from datetime import datetime

from score_db import score_db_base

import warnings

def extract_unique_stats(strings):
    # Create sets to store unique values in the second and last positions
    first_position_set = set()
    second_position_set = set()
    third_position_set = set()
    last_position_set = set()
    
    # Iterate over the set of strings
    for s in strings:
        parts = s.split('_')
        
        # Add the second and last elements to their respective sets
        if len(parts) > 1: #and parts[-1] != 'None':  # Ensure there are at least 2 parts, 
            first_position_set.add("_".join(parts[0:1]))
            second_position_set.add('_'.join(parts[1:2]))
            third_position_set.add("_".join(parts[2:3]))
            last_position_set.add(parts[-1])
    
    # Convert sets to sorted lists to maintain a consistent order
    first_position_list = sorted(list(first_position_set))
    second_position_list = sorted(list(second_position_set))
    third_position_list = sorted(list(third_position_set))
    last_position_list = sorted(list(last_position_set))
    
    # Combine the two lists into a 2D array (list of lists)
    unique_positions = [first_position_list, second_position_list, third_position_list, last_position_list]
    
    return unique_positions

def get_data_frame(experiment_list,
                   metric_list,
                   region='global',
                   start_date='1979-01-01 00:00:00',
                   stop_date='2026-01-01 00:00:00'):
    """request from the score-db application experiment data
    Database requests are submitted via score-db with a request dictionary
    """
    
    request_dict = {
        'db_request_name': 'expt_metrics',
        'method': 'GET',
        'params': {
            'datestr_format': '%Y-%m-%d %H:%M:%S',
            'filters': {
                'experiment':{
                    'name': {
                        'exact': experiment_list
                    },
                },
                'metric_types': {
                    'name': {
                        'exact': metric_list
                    },
                },
                'regions': {
                    'name': {
                        'exact': [region]
                    }
                },
                'time_valid': {
                    'from': start_date,
                    'to': stop_date
                }
            },
            'ordering': [{'name': 'time_valid', 'order_by': 'asc'}]
        }
    }

    db_action_response = score_db_base.handle_request(request_dict)    
    data_frame = db_action_response.details['records']
    
    # sort by timestamp, created at
    data_frame.sort_values(by=['expt_name',
                               'name',
                               'region_name',
                               'time_valid',
                               'usage',
                               'created_at'], 
                               inplace=True)

    # remove duplicate data
    data_frame.drop_duplicates(subset=['expt_name',
                                       'name',
                                       'region_name',
                                       'time_valid',
                                       'usage'], 
                               keep='last', inplace=True)
    
    return data_frame

class SOCADiagsTimeSeries(object):
    def __init__(self,
                 start_date,
                 stop_date,
                 data_frame=None,
                 input_data_frame=False,
                 experiment_name = '3dvar_coupledreanl_scoutrun_1979streamv1_test',
                 select_metric_types = True,
                 metric_types='%',
                 experiment_id=None,
                ):
        """Download metrics data for given experiment name
        """
        self.init_datetime = datetime.now()
        self.experiment_name = experiment_name
        self.select_metric_types = select_metric_types
        self.metric_types = metric_types
        self.experiment_id = experiment_id
        if input_data_frame and type(self.metric_types) == str:
            self.data_frame = data_frame[(
                data_frame['expt_name'] == self.experiment_name) &            
                (data_frame['name'] == self.metric_types)]
        
        else:
            self.data_frame = get_data_frame(
                [self.experiment_name],
                [self.metric_types],
                start_date=start_date,
                stop_date=stop_date)
        
    def build(self, qc_threshold=0):
        self.unique_stat_list = extract_unique_stats(
                                            set(self.data_frame['name']))
        
        self.timestamp_dict = dict()
        self.value_dict = dict()
        for i, stat_name in enumerate(self.unique_stat_list[0]):
            for j, var_name in enumerate(self.unique_stat_list[1]):
                for k, soca_stage in enumerate(self.unique_stat_list[3]):
                    self.timestamp_dict[f'{stat_name}_{var_name}_{soca_stage}'] = dict()
                    self.value_dict[f'{stat_name}_{var_name}_{soca_stage}'] = dict()
                
        for key in self.value_dict.keys():
            for instrument_name in set(self.data_frame.metric_instrument_name):
                    sensor_label = f'{instrument_name}'
                    self.timestamp_dict[key][sensor_label] = list()
                    self.value_dict[key][sensor_label] = list()        
        
        self.sensorlabel_dict = dict()
        yval = 0            
        
        if qc_threshold is None:
            usage_match = 'noqc'
        elif qc_threshold == 0:
            usage_match = 'effectiveQC_eq_0'
        else:
            usage_match = 'effectiveQC_lt_x'
        
        for row in self.data_frame.itertuples():
            metric_name_parts = row.name.split('_')
            if metric_name_parts[2] == row.metric_instrument_name and row.expt_name == self.experiment_name and row.usage == usage_match:
                #TODO: add support for nonzero QC values
                
                stat_name = '_'.join(metric_name_parts[0:2])
                soca_stage = metric_name_parts[-1]
                
                stat_label = f'{stat_name}_{soca_stage}'
                
                sensor_label = f'{row.metric_instrument_name}'
                timestamp = row.time_valid#.timestamp()
                
                value = row.value

                #print(gsi_stage, stat_name, sensor_label, time_label, value)
                self.timestamp_dict[stat_label][sensor_label].append(timestamp)
                #self.timelabel_dict[stat_label][sensor_label].append(time_label)
                self.value_dict[stat_label][sensor_label].append(value)
                
                if not sensor_label in self.sensorlabel_dict.keys():
                    self.sensorlabel_dict[sensor_label] = yval
                    yval -= 1
        
                #print(gsi_stage, stat_name, sensor_label, self.sensorlabel_dict[sensor_label])
                
            elif usage_match == 'effectiveQC_lt_x':
                warning.warm('plotting not supported for nonzero QC thresholds')
        
    def print_init_time(self):
        print("SOCADiagsTimeSeries object init date and time: ",
              f"{self.init_datetime}")