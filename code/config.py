import logging
from click import option
import yaml
import os
from pathlib import Path

from models import model


logging.basicConfig( level = logging.INFO )

class Configuration: 
    def __init__(self, yaml_file, model_class='mdl'):
        '''
        :param model_class - choices - ['mdl', 'tts']
        '''
        
        self.observation_window = (500, 1500)
        self.datasets = []
        
        ## setup model class 
        if model_class == 'mdl':
            self.model_class = model.MdlModel 
        elif model_class == 'tts':
            self.model_class = model.TrainTestModel
        else:
            raise ValueError('Model class \'{}\' unknown '.format(model_class))
        
        self._load_config_from_yaml(yaml_file)

      


    def _load_config_from_yaml(self, file):
        logging.info("Reading yaml file {}".format(file))
        config = self._read_yaml(file)
        ## Check our configurations for datasets 
        self.data_dir = config.get('data_dir', '../data')
        self.observation_window = self._read_observation_window(config) or self._observation_window
        self.max_number_parents = config.get('max_number_parents', 3)
        self.markov_order = config.get('markov_order', 10)

        self.n_trials = config.get('n_trials', 40)
        self.time_per_trial = config.get('time_per_trial', 2000)
        self.min_active_window = config.get('min_active_window', 100)
        self.stimuli = config.get('stimuli', [])
        self.n_recordings = config.get('n_recordings', 1000000)
        
        self._lazyload_datasets(config)
        logging.info("Loaded datasets: {}".format(self.datasets))

        
    def _lazyload_datasets(self, config ):
        dataset_paths = config['datasets']
        if dataset_paths:
            ## Construct datasets objects
            for p in dataset_paths:
                self.datasets.append(Dataset(p, self.data_dir, train_test_split=self.model_class == model.TrainTestModel))
        
            
        
    def _read_observation_window(self, config): 
        window = config['observation_window']
        if window:
            return tuple(map(int, window.split(',')))
    
        return None
    
    
    def _read_yaml(self, file):
        try:
            with open(file, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            logging.error("Reading yaml file {} failed. File not found".format(file))
            raise 

    def observation_window_string(self):
        return '_'.join(map(str, self.observation_window))



TRAIN_TEST_SPLIT = 'train_test_split' ## directory prefix  


class Dataset:
    def __init__(self, path, data_dir, train_test_split=False):
        self.path = path
        self.data_dir = data_dir
        self.preprocess = not os.path.exists(self.preprocessed_path)

        ## TODO: think of filetype - feather 


        self.train_test_split = train_test_split 


    @property     
    def filtered_spikes_path(self):
        return '{}/filtered_spikes/{}'.format(self.data_dir, self.name)

    
    def filtered_spikes_with_stimulus_path(self, stimulus_id):
        return '{}/filtered_spikes/{}/{}'.format(self.data_dir, self.name, stimulus_id)

    @property
    def preprocessed_path(self):
        ## Save the preprocessed dataset in the parent directory 
        name = Path(self.path).name
        # return '{}/../{}'.format(self.data_dir, name)
        return '{}/../../{}'.format(self.data_dir, name)

    @property
    def unit_to_region_path(self):
        return '{}/{}_unit_to_region.pkl'.format(self.data_dir, self.name) 

    @property
    def name(self):
        return Path(self.path).stem

    @property
    def recording_to_units(self):
        return '{}/{}_recording_to_units.pkl'.format(self.data_dir, self.name) 

    def active_times_path(self, time):
        return '{}/active_times/{}_active_times_{}.pkl'.format(self.data_dir, self.name, time)
    
    def active_times_with_stimulus_path(self, time, stimulus_id):
        return '{}/active_times/{}/stimulus_{}_active_times_{}.pkl'.format(self.data_dir, self.name, stimulus_id, time)

    def spiking_rates_path(self, time):
        return '{}/spiking_rates/{}_spiking_rates_{}.pkl'.format(self.data_dir, self.name, time)

    def DI_values_path(self, time, stimulus_id=None):
    
        prefix = '{}/{}'.format(self.data_dir, 'tts') if self.train_test_split else self.data_dir

        if stimulus_id is not None:
            return '{}/DI_values/{}/{}/{}'.format(prefix, self.name, stimulus_id, time)
        return '{}/DI_values/{}/{}'.format(prefix, self.name, time)


   
    def best_parent_path(self, time, options, stimulus_id=None, raw_DI=False):
        '''
        Path to directory where the best parentsets for each units are stored 
        '''
        prefix = '{}/{}'.format(self.data_dir, 'tts') if self.train_test_split else self.data_dir
        if raw_DI:
            if stimulus_id is not None:
                return '{}/best_raw_parents/{}/{}/{}'.format(prefix, self.name, stimulus_id, time, options)    
            return '{}/best_raw_parents/{}/{}'.format(prefix, self.name, time, options)

        if stimulus_id is not None:
            return '{}/best_parents/{}/{}/{}'.format(prefix, self.name, stimulus_id, time, options)    
        return '{}/best_parents/{}/{}'.format(prefix, self.name, time, options)   
    

    def aggregate_best_parent_filepath(self, time, options, stimulus_id=None, raw_DI=False):
        '''
        Path to the aggregate parentsets 
        '''
        prefix = '{}/{}'.format(self.data_dir, 'tts') if self.train_test_split else self.data_dir
        if raw_DI:
            if stimulus_id is not None:
                return '{}/best_raw_parents/{}/{}/{}/best_parents_{}.pkl'.format(prefix, self.name, stimulus_id, time, options)
            return '{}/best_raw_parents/{}/{}/best_parents_{}.pkl'.format(prefix, self.name, time, options)            

        if stimulus_id is not None:
            return '{}/best_parents/{}/{}/{}/best_parents_{}.pkl'.format(prefix, self.name, stimulus_id, time, options)
        return '{}/best_parents/{}/{}/best_parents_{}.pkl'.format(prefix, self.name, time, options)
    

    def connection_strength_filepath(self, time, options, stimulus_id=None, raw_DI=False):
        prefix = '{}/{}'.format(self.data_dir, 'tts') if self.train_test_split else self.data_dir

        if raw_DI:
            if stimulus_id is not None:
                return '{}/raw_group_connectivity/{}/{}/{}/connectivity_{}_{}.pkl'.format(prefix, self.name, stimulus_id, time, self.name, options)
            return '{}/raw_group_connectivity/{}/{}/connectivity_{}_{}.pkl'.format(prefix, self.name, time, self.name, options)


        if stimulus_id is not None:
            return '{}/group_connectivity/{}/{}/{}/connectivity_{}_{}.pkl'.format(prefix, self.name, stimulus_id, time, self.name, options)
        return '{}/group_connectivity/{}/{}/connectivity_{}_{}.pkl'.format(prefix, self.name, time, self.name, options)

    def connection_strength_plot_filepath(self, time, options, stimulus_id=None, raw_DI=False):
        prefix = '{}/{}'.format(self.data_dir, 'tts') if self.train_test_split else self.data_dir
        if raw_DI:
            if stimulus_id is not None:
                return '{}/raw_group_connectivity/{}/{}/{}/connectivity_{}_{}.png'.format(prefix, self.name, stimulus_id, time, self.name, options)
            return '{}/raw_group_connectivity/{}/{}/connectivity_{}_{}.png'.format(prefix, self.name, time, self.name, options)

        if stimulus_id is not None:
            return '{}/group_connectivity/{}/{}/{}/connectivity_{}_{}.png'.format(prefix, self.name, stimulus_id, time, self.name, options)
        return '{}/group_connectivity/{}/{}/connectivity_{}_{}.png'.format(prefix, self.name, time, self.name, options)

    def __repr__(self):
        return self.path