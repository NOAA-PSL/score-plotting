from datetime import datetime

from score_db import score_db_base

def get_data_frame(experiment_list,
                   metric_list,
                   start_date='1979-01-01 00:00:00',
                   stop_date='2026-01-01 00:00:00',
                   array=False,
                   ):
    """request from the score-db application experiment data
    Database requests are submitted via score-db with a request dictionary
    """
    
    if array:
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
            'name': {'exact': metric_list}
        }
    else:
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
                        'rgs_name': {
                            'exact': ['global']
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

    if array:
        # sort by timestamp, created at
        data_frame.sort_values(by=['expt_name',
                                   'metric_name',
                                   'assimilated',
                                   'time_valid',
                                   'created_at'], 
                                   inplace=True)

        # remove duplicate data
        data_frame.drop_duplicates(subset=['expt_name',
                                           'ensemble_member',
                                           'metric_name',
                                           'assimilated',
                                           'time_valid'], 
                                   keep='last', inplace=True)
    else:
        # sort by timestamp, created at
        data_frame.sort_values(by=['expt_name',
                                   'ensemble_member',
                                   'name',
                                   'usage',
                                   'time_valid',
                                   'created_at'], 
                                   inplace=True)

        # remove duplicate data
        data_frame.drop_duplicates(subset=['expt_name',
                                           'ensemble_member',
                                           'name',
                                           'usage',
                                           'time_valid',
                                           ], 
                                   keep='last', inplace=True)

    return data_frame

class GSIConvTimeSeries(object):
    def __init__(self,
                 start_date,
                 stop_date,
                 data_frame=None,
                 input_data_frame=False,
                 experiment_name = '3dvar_coupledreanl_scoutrun_1979streamv1_test',
                 select_metric_types = True,
                 metric_types='%',
                 experiment_id=None,
                 array=False
                ):
        """Download metrics data for given experiment name
        """
        self.init_datetime = datetime.now()
        self.experiment_name = experiment_name
        self.select_metric_types = select_metric_types
        self.metric_types = metric_types
        self.experiment_id = experiment_id
        self.array = array
        if self.array:
            metric_name_key = 'metric_name'
        else:
            metric_name_key = 'name'
        
        if input_data_frame and type(self.metric_types) == str:
            self.data_frame = data_frame[(
                data_frame['expt_name'] == self.experiment_name) &            
                (data_frame[metric_name_key] == self.metric_types)]
        
        else:
            self.data_frame = get_data_frame(
                [self.experiment_name],
                [self.metric_types],
                start_date=start_date,
                stop_date=stop_date,
                array=self.array)
        
    def build(self):
        
        self.timestamp_dict = dict()
        self.value_dict = dict()
        self.pressure_levs_dict = dict()

        for metric in [self.metric_types]:
            self.timestamp_dict[metric] = dict()
            self.value_dict[metric] = dict()
            self.pressure_levs_dict[metric] = dict()
                
        for key in self.value_dict.keys():
            for instrument_name in set(self.data_frame.metric_instrument_name):
                if instrument_name is not None:
                    self.timestamp_dict[key][instrument_name] = dict()
                    self.value_dict[key][instrument_name] = dict()
                    self.pressure_levs_dict[key][instrument_name] = dict()
                    
                    for usage in ['asm', 'mon', 'rej']:
                        self.timestamp_dict[key][instrument_name][usage] = list()
                        self.value_dict[key][instrument_name][usage] = list()
                        self.pressure_levs_dict[key][instrument_name][usage] = dict()
        
        for row in self.data_frame.itertuples():
            if row.expt_name == self.experiment_name:
                if self.array:
                    row_metric_name = row.metric_name
                    
                    if (row.array_coord_labels[0] in self.pressure_levs_dict[key][instrument_name][row.usage].keys() and
                        self.pressure_levs_dict[key][instrument_name][row.usage][row.array_coord_labels[0]] != row.array_index_values[0]):
                        raise ValueError(f"{row.array_coord_labels[0]} mismatch encountered during timeseries creation")
                    if (row.array_coord_labels[1] in self.pressure_levs_dict[key][instrument_name][row.usage].keys() and
                        self.pressure_levs_dict[key][instrument_name][row.usage][row.array_coord_labels[1]] != row.array_index_values[1]):
                        raise ValueError(f"{row.array_coord_labels[1]} mismatch encountered during timeseries creation")
                    
                    self.pressure_levs_dict[key][instrument_name][row.usage][row.array_coord_labels[0]] = row.array_index_values[0]
                    self.pressure_levs_dict[key][instrument_name][row.usage][row.array_coord_labels[1]] = row.array_index_values[1]
                else:
                    row_metric_name = row.name
                
                self.timestamp_dict[row_metric_name][row.metric_instrument_name][row.usage].append(row.time_valid)
                self.value_dict[row_metric_name][row.metric_instrument_name][row.usage].append(row.value)
        
    def print_init_time(self):
        print("GSIConvTimeSeries object init date and time: ",
              f"{self.init_datetime}")