# -*- coding: utf-8 -*-
"""

this determines what times (out of entire time-series) to use; issue being fitting can get distorted if there are large times of inactivity (all zeros)



@author: cjquinn, yididiya
"""

import os
import argparse
import numpy as np
import pandas
from multiprocessing import Pool, cpu_count


from config import Configuration 
from utils import read_pickle_file, pickle_object




class ActiveTimes():

    def __init__(self, config, dataset, stimulus_id=None):
        self.config = config
        self.dataset = dataset

        ## Now, we are looking at stimuli id 
        self.stimulus_id = stimulus_id
        
        self.filtered_spikes_dir = self.dataset.filtered_spikes_path
        
        if self.stimulus_id is not None: 
            self.filtered_spikes_dir = self.dataset.filtered_spikes_with_stimulus_path(self.stimulus_id)
        
        self.observation_window = self.config.observation_window
    
    def find_active_times(self):
        '''
        :param filtered_spikes_dir - directory where the filtered spikes resides 
        :param data_dir - where our intermediate data files are located  
        '''

        if self.check_active_times():
            ## Save time
            return
        n_trials = self.config.n_trials
        time_per_trial = self.config.time_per_trial
        min_active_window = self.config.min_active_window
        markov_order = self.config.markov_order 

        

        active_time_dict= {'min_active_window':min_active_window, 'markov':markov_order} #this will hold the active times
        
        
        max_time = n_trials * time_per_trial ## experiment timespan in milliseconds
        
        

        ## Load recordings to units pickle
        recording_units = read_pickle_file(self.dataset.recording_to_units)

        
        for recording in recording_units.keys():
            
            ## Get units in the recording 
            units = recording_units[recording]
            
            #create a vector counting number of cells active at each time
            active_times = np.zeros((max_time,1))

            for unit in units:
                prefiltered_spike_train = read_pickle_file(self.filtered_spikes_dir + os.sep + unit + '.pkl')
                if not prefiltered_spike_train: 
                    print('LOG: unit {} not prefiltered yet '.format(unit))
                    continue

                
                #check both spikes in spike_vec and past_filt; add together
                tmp = np.sum(prefiltered_spike_train['past_filt'],1)
                tmp = tmp.reshape(tmp.shape[0],1)
                tmp += prefiltered_spike_train['spike_vec']                    

                active_times += tmp #spikes_filt['spike_vec']
            
            times_for_fitting = np.zeros(np.shape(active_times))
            
            ## Decide times for fitting 
            ## Changing it to two optimized for loops may help 
            for t in range(np.size(times_for_fitting)):
                
                sliced_t = t % time_per_trial
                if sliced_t >= self.observation_window[0] and sliced_t <= self.observation_window[1] and sliced_t > markov_order:
                    min_t = max(0, t-min_active_window)

                    ## May be the idea of active may vary from experiment to experiment 
                    if np.sum(active_times[min_t: t+1]) > 0: 
                        times_for_fitting[t] = 1
                        
        
            times_for_fitting = times_for_fitting.astype(np.bool_)

            
            active_time_dict[recording] = times_for_fitting
        
        self.save_active_times(active_time_dict)


    def check_active_times(self):
        '''
        Checks if active files exists
        '''
        ## Check where to filter with stimulus or not
        if self.stimulus_id is not None:
            path = self.dataset.active_times_with_stimulus_path('_'.join(map(str, self.observation_window)), self.stimulus_id)
        else:
            path = self.dataset.active_times_path('_'.join(map(str, self.observation_window)))

        if os.path.exists(path):
            return path
        
        return None


    def save_active_times(self, active_time_dict):
        
        ## Check where to filter with stimulus or not
        if self.stimulus_id is not None:
            pickle_object(active_time_dict, self.dataset.active_times_with_stimulus_path('_'.join(map(str, self.observation_window)), self.stimulus_id))
        else:
            pickle_object(active_time_dict, self.dataset.active_times_path('_'.join(map(str, self.observation_window))))




def find_active_times_parallel(params):
    config, dataset, stimulus_id = params
    ActiveTimes(config, dataset, stimulus_id=int(stimulus_id)).find_active_times()


def main(config):
    for dataset in config.datasets:
        data = pandas.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or data.stimn.unique() 


        # ## Serial 
        for stimulus_id in stimuli:
            print('Finding active times for {}'.format(stimulus_id))
            ActiveTimes(config, dataset, stimulus_id=int(stimulus_id)).find_active_times()


        ## Parallel 
        # with Pool(max(1,cpu_count()-1)) as pool:
        #     print('In the pool now')
        #     # result = pool.starmap(prefilter_unit, input_generator() )
        #     pool.map(find_active_times_parallel, [(config, dataset, stimulus_id) for stimulus_id in stimuli])
        #     print('outta the pool now')

        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find active times ')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml',)

    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config)
    
    main(conf)    