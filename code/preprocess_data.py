#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:12:06 2022

@author: cjquinn, yididiya 
"""

#from click import confirm

import os
from pathlib import Path
import pickle

import pandas as pd
from config import Configuration

import argparse




'''
Preprocessing
'''

def get_recording_details(id_, index): 
    '''
    This function extracts the information on the "id" column into a tuple. 
    
    (mouse_id, hemisphere, pre, region, unit_id )
    
    Eg. get_recording_details('ET012_left_pre_LM_1') = ('ET012', 'left', 'pre', 'LM', '1')
    
    '''
    
    recording_info = id_.split('_')
    #assert(len(recording_info) == 5 and index < 5) 
    
    return recording_info[index]


def preprocess_dataset(data, config):
    '''
    Function expands the information in the id column 
    :param data - pandas object 
    '''

    ## Dont preprocess twice 
    if 'mouse_id' in data.columns:
        print('Skipping dataset preprocessing. Already done.')
        return data

    ## remove negative times values 
    data = data[data.times >= 0]
    
    data['mouse_id'] = data['id'].apply(lambda id_: get_recording_details(id_, 0))
    data['hemisphere'] = data['id'].apply(lambda id_: get_recording_details(id_, 1))
    data['experiment'] = data['id'].apply(lambda id_: get_recording_details(id_, 2))
    data['region'] = data['id'].apply(lambda id_: get_recording_details(id_, 3))
    data['recording'] = data['mouse_id'] + '_' + data['hemisphere'] + '_' + data['experiment']
    
    ## Update spike times 
    ## In the datset the spike times are considered after for just trial. 
    
    data['times'] = data['times'] + data['trial'] * config.time_per_trial / 1000.0
    
    data.times = data.times * 1000. ## spike times in ms

    data.reset_index(drop=True, inplace=True)

    ## Assertions 
    assert data.times.max() <= config.time_per_trial * len(data.trial.unique()), "Preprocessing went terribly wrong, CHECK times conversion"
    assert data.times.min() >= 0, "Negative times values"

    return data





def main(config): 
    for dataset in config.datasets:
        print("Working with dataset " + dataset.name)
        
        ## prep output directory
        if not os.path.exists(dataset.data_dir): os.mkdir(dataset.data_dir)
        
        ## preprocess dataset or load  if already preprocessed
        if dataset.preprocess:    
            ## Read dataset
            data = pd.read_feather(dataset.path)
            data = preprocess_dataset(data, config)
            
            data.to_feather(dataset.preprocessed_path)
        else:
            data = pd.read_feather(dataset.preprocessed_path)

        assert 'recording' in data.columns, 'Dataset is not preprocessed'
        units = data.id.unique()
        recordings = data.recording.unique()

        
        ## Write recordings mapped to their set of units under that recording 
        ## This is equivalent to listcells in the previous implementation 
        with open(dataset.recording_to_units, 'wb') as f:    
            recording_units  =  {}

            for recording in recordings:
                recording_units[recording] = list(data[data.recording == recording].id.unique())
            pickle.dump(recording_units, f)
        

        ## Write unit to region mappings
        with open(dataset.unit_to_region_path, 'wb') as f:
            unit_region = {}
            for unit in units: 
                unit_region[unit] = unit.split('_')[-2]
            pickle.dump(unit_region, f)        

        ## Assertions 
        assert os.path.exists(dataset.unit_to_region_path), "Unit-to-region file is not saved properly"
        assert os.path.exists(dataset.unit_to_region_path), "Recording-to-units file is not saved properly"

        print("DONE")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess and generating necessary pickles')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml',
                    help='sum the integers (default: find the max)')

    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config)
    main(conf)    
    
    
    