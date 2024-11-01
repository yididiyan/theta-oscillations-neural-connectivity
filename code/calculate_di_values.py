# -*- coding: utf-8 -*-
"""


This code will search for the parents of a single cell, get the DI for all individual, pair, and triplet of parents.

It does so separately for different time intervals and types of restrictions on parent sets.  

No need to parallelize -- statsmodel will do so while fitting, and getting multiprocessor to work can be hard
-- instead, just use sbatch --array to run many instances of the job

@author: cjquinn
"""

import argparse
import logging 
        

import numpy as np

import pandas
from utils import pickle_object, read_pickle_file, timer_func, profileit

from config import Configuration

from time import time


from search_parents_optimized import SearchParentSet






def read_best_parentset(filename):
    '''
    read recording/unit pairs in a best_parent set 
    '''
    print('Reading best parent set:  {}'.format(filename))
    
    best_pars = read_pickle_file(filename)
    

    Yunits = list(best_pars.keys())
    Xunits = list(set([p for d in best_pars.values() for p in d['pars']]))
    all_units = list(set(Xunits + Yunits))
    recordings = set([ '_'.join(y.split('_')[:-2]) for y in  Yunits])

    recording_units = {k:[] for k in recordings}
    for u in all_units:
        recording_units['_'.join(u.split('_')[:-2])].append(u)


    unit_pars = {k:list(v['pars']) for k,v in best_pars.items()}

    return all_units, Yunits, Xunits, recording_units, unit_pars
  


class CalculateDI(SearchParentSet):

    def __init__(self, config, dataset, data, stimulus_id=None, part_list=['all2all'], best_pars_file=None):


        self.use_only_best_pars = False 
        ## load best parents set 
        if best_pars_file:
            self.use_only_best_pars = True
            _, self.Yunits, self.Xunits, self.recording_units, self.unit_pars = read_best_parentset(best_pars_file)

        super(CalculateDI, self).__init__(config, dataset, data, stimulus_id, part_list )


        ## Sanity check 
        ## recordings we're looking at match with pandas recordings 
        assert set(self.data.recording.unique() ) == set(self.get_list_of_recordings()), "FATAL: Recordings mismatch with pandas "
        
            

        
    def get_list_of_recordings(self):
        ## Get list of all units 
        return self.recording_units.keys() if self.use_only_best_pars else self.data.recording.unique() 

    def get_list_of_units(self):
        return self.Yunits if self.use_only_best_pars else list(self.data.id.unique())
        

    def get_units_in_recording(self, rec_id, unit_id=None):
        if self.use_only_best_pars:
            ## select only the units in the best parent set of "unit_id"
            ## don't bother loading every non-best unit  
            assert unit_id is not None, "Fetching parent units requires \"unit_id\" to be provided "
            return self.unit_pars[unit_id] 
        else:
            return list(self.data[self.data.recording == rec_id].id.unique())




    @property
    def filtered_spikes_path(self):
        return self.dataset.filtered_spikes_with_stimulus_path(self.stimulus_id)


    @property
    def data_dir(self):
        return self.dataset.data_dir 


        
def main(args):
    
    
    ## Pull configuration 
    config = Configuration(args.config, args.model)
    best_parentset = args.best_parentset 

    
    for dataset in config.datasets:
    # for dataset, prefiltered_dataset in zip(datasets, prefiltered_datasets):
        
        ## Load the data 
        
        print('Loading dataset {}'.format(dataset))
        data = pandas.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or list(data.stimn.unique())

        # random.shuffle(stimuli)
        for stimulus_id in stimuli:
            CalculateDI(config, dataset, data, stimulus_id=int(stimulus_id), best_pars_file=best_parentset)()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for parent sets')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml')
    parser.add_argument('--model', action='store', type=str, default='mdl')
    parser.add_argument('--best-parentset', action='store', type=str, default='')

    args = parser.parse_args()
    
    
    main(args)
