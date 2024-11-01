# -*- coding: utf-8 -*-
"""

This file preprocesses data to speed up regression.  Specifically, it creates columns of exogeneous variables that will later get used in fitting other units.

------------------

Instead of doing an exhaustive search over Markov parameters, we will specify a single mapping of past values to vector of variables.  This will be done ahead of time in  this file prefilter_data.py

Eg   (X_t-1 + X_t-2)  (X_t-3+ X_t-4+ X_t-5)  (X_t-6 .. X_t-10)

Running as a serial process (but submitting many independent ones) may avoid problems of freezing up without error when using multiprocessing

@author: cjquinn, yididiya 
"""
import os
import argparse
import logging
import random

from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pickle

from config import Configuration

logging.basicConfig( level = logging.INFO )


class Prefilter():

    def __init__(self, data, config, output_dir, stimulus_id=None, update=True):
        '''
        :param update - if True it will only do prefiltering for the units not in the file 
        '''
        self.data = data
        self.config = config
        self.output_dir = output_dir
        self.units = self.data.id.unique()

        # stimulus_id - stimuli id is used to slice the dataset based on stimuli ids 
        self.stimulus_id = stimulus_id
        if self.stimulus_id is not None:
            ## More filtering with stimuli id 
            self.units = self.data[self.data.stimn == self.stimulus_id].id.unique()
        
        if update:
            self.units = [ u for u in self.units if not os.path.exists(self.get_filtered_spikes_filename(u))]

    def apply_filter(self, times, filt, max_time):
        """
        # :param unit - the unit whose spikes we are filtering
        :param times - the spike times of the unit 
        :param filtr - filtering mask, eg. how far back we look and more 
        :param max_time - time span of the entire experiment 

        This code applies a filter so when want to condition on the past of this cell, instead of having separate variables for each ms interval, we will combine so only 3 variables for past 10 ms.
        """
        times = np.array(times)
        # times = np.ceil(times*1000) #from s to ms
        times = times.astype(int) 
        
        times = times[times<max_time]
        
        #make a full, 1-d binary array of spikes
        spike_vec = np.zeros((max_time,1))
        spike_vec[times] += 1
        
        #initialize the array to store the filtered past
        filt_past = np.zeros((max_time,filt.shape[1]))
        
        for t in range(filt.shape[0]): 
            #put NaN at start so know don't have enough past yet to use for modeling
            filt_past[t,:] = np.NaN
        
        for t in range(filt.shape[0],max_time):
            #the way we set up filt [1 0 0; 1 0 0; 0 1 0; 0 1 0; ...] we want to premultiply by a row vector in reverse time order, so [ X_{t-1} X){t-2} X_{t-3} ...   ]
        
            filt_past[t,:] = np.dot(np.transpose(spike_vec[range(t-1,t-filt.shape[0]-1,-1)]),filt)

        ## Assertions 
        markov_order = filt.shape[0]
        assert np.sum(filt_past[markov_order:, 0]) + np.sum(filt_past[markov_order:, 1]) == np.sum(filt_past[markov_order:, 2])   , "Error prefilting past of unit"
        
        return spike_vec, filt_past
            






    def prefilter_unit(self, data, unit):
        '''
        :param data - dataset in pandas format
        :param unit - cell/unit id  
        :param: output_dir - directory where the prefiltering output will be stored 
        '''
        logging.info("Working with unit " + unit)
        if os.path.exists(self.get_filtered_spikes_filename(unit)):
            logging.info("Skipping prefiltering; already prefiltered...")
            return True

        
        max_time = self.config.n_trials * self.config.time_per_trial ## 40 trials, each 2000 ms long 
        markov_order = self.config.markov_order


        spikes_filt = {}   
        
        filt = np.zeros((3,markov_order))  #the shape is (a,b) where b is the limit of how far back in time we consider and a is the number of variables we want to make out of the past b ms.
        
        #for own past, model refractory period separate from further past    
        filt[0,0:2] += 1
        filt[1,2:markov_order] += 1
        
        #when this cell is an exogeneous variable, use whole past 10ms
        filt[2,:] += 1

        filt=np.transpose(filt)  
        

        
        tmp_vec, tmp_past = self.apply_filter(data.times, filt, max_time)
        
        
        #to save on storage, speed, cast as uint8 
        if np.max(tmp_past)>255:
            print('\n\n ERROR -- don"t save as uint8 \n\n\n')
            
        tmp_vec = tmp_vec.astype(np.uint8)
        tmp_past = tmp_past.astype(np.uint8)
        
        spikes_filt['spike_vec'] = tmp_vec
        spikes_filt['past_filt'] = tmp_past


        #now save it
        with open(self.get_filtered_spikes_filename(unit), 'wb') as f:
            pickle.dump(spikes_filt, f)


        print("LOG: prefiltering {} complete".format(unit))
        return True

    def prefilter_unit_(self, unit):
        data = self.data[self.data['id'] == unit]
        
        ## Filter the data with simuli_id if it exists 
        if self.stimulus_id is not None:
            data = data[data['stimn'] == self.stimulus_id]
        return self.prefilter_unit(data, unit) 

    # def prefilter_input_generator(self):
    #     for unit in self.units:
    #         yield (self.data[self.data['id'] == unit], unit, self.output_dir, )

    def prefilter(self): 
        ''''
        Preprocess spike train for unit and persist it
        :param data - dataset in pandas dataframe format
        :param: output_dir - directory where the prefiltering output will be stored 
        '''

        ## Update spike times 
        ## In the datset the spike times are considered after for just trial. 
        



        
        # def prefilter_unit_singular(unit):
        #     prefilter_unit(self.data[self.data['id'] == unit], unit, output_dir) 
        
        random.seed(os.urandom(1000))
        random.shuffle(self.units)
        list(map(self.prefilter_unit_, self.units))
        

        # with Pool(max(1,cpu_count()-1)) as pool:
        #     print('In the pool now')
        #     # result = pool.starmap(prefilter_unit, input_generator() )
        #     result = pool.map(self.prefilter_unit_, self.units)
        #     print('outta the pool now')
        #     if result: print('Doing well')

    def get_filtered_spikes_filename(self, unit_id):
        '''
        :param unit_id - the identifier of the unit 
        :param filtered_spikes_dir - directory where our prefiltered spikes are located 
        '''
        return self.output_dir + os.sep + unit_id + '.pkl'



def read_dataset(dataset):
    ## prep output directory
    if not os.path.exists(dataset.data_dir): os.mkdir(dataset.data_dir)


    if os.path.exists(dataset.preprocessed_path):    
        ## Read dataset
        data = pd.read_feather(dataset.preprocessed_path)
    else: 
        data = pd.read_feather(dataset.path)

    return data


def create_prefilter_object(config, dataset, data, stimulus_id):
    
    filtered_spikes_dir = dataset.filtered_spikes_with_stimulus_path(int(stimulus_id))

    os.makedirs(filtered_spikes_dir, exist_ok=True)
    
    return Prefilter(data, config, filtered_spikes_dir, stimulus_id=int(stimulus_id))
    

def main(config): 
    for dataset in config.datasets:
    
        data = read_dataset(dataset)

        if config.stimuli: 
            stimuli = config.stimuli
        else:
            ## Shuffle 
            stimuli = data.stimn.unique()
            random.seed(os.urandom(1000))
            random.shuffle(stimuli)

        for stimulus_id in stimuli: 
            prefilter = create_prefilter_object(config, dataset, data, stimulus_id)
            prefilter.prefilter()
        

 

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prefilter spike datasets ')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml',
                    help='')

    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config)
    
    main(conf)    
