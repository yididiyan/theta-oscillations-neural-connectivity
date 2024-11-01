# -*- coding: utf-8 -*-
"""


This code will search for the parents of a single cell, get the DI for all individual, pair, and triplet of parents.

It does so separately for different time intervals and types of restrictions on parent sets.  

No need to parallelize -- statsmodel will do so while fitting, and getting multiprocessor to work can be hard
-- instead, just use sbatch --array to run many instances of the job

@author: cjquinn, yididiya
"""

import os
import random
import argparse
import glob 
from pathlib import Path
        
from matplotlib.pyplot import get

import numpy as np

import statsmodels.api as sm
import pandas
from utils import pickle_object, read_pickle_file, timer_func, profileit

from config import Configuration

from time import time

from models import model
import logging


logging.basicConfig( level = logging.INFO )


  
  

class SearchParentSet():

    def __init__(self, config, dataset, data, stimulus_id=None, part_list=['all2all']):
        self.config = config 
        self.dataset = dataset 
        self.data = data
        self.part_list = part_list
        self.stimulus_id = stimulus_id 
        

        ## Observation window 
        self.times_list = ['_'.join(map(str, config.observation_window))]

        self.active_times = {}

        # This variable cached the set of Xvalues of potential parents so that we don't load them 
        # This cache only works for a recording and overriden when working with unit with another recording
        self.past_filter_cache = {
            'recording': None,  
            'Past': None,
            'units': None
        }

        assert self.stimulus_id, "Stimulus id is required to search for parents"
        ## Slice data by stimulus_id 
        self.data = self.data[self.data.stimn == stimulus_id]


        ## slice data by n_recordings 
        
        recs = sorted(self.get_list_of_recordings())
        print('Working with recordings:  ', ', '.join(recs))
        self.data = self.data[self.data.recording.isin(recs[:config.n_recordings])]
        
        self.model_class = self.config.model_class
        logging.info('Using model class ' + str(self.model_class))
        
    

    def get_list_of_recordings(self):
        ## Get list of all units 
        return self.data.recording.unique()

    def get_list_of_units(self):
        '''
        Returns list of units in the datasets
        Note: only selects prefiltered units. Rationale: if no prefilter, units was too quiet. 
        '''
        return list(self.data.id.unique())
        # ## Get list of all units 
        # if self.stimulus_id is not None:
        #     return [Path(f).stem for f in glob.glob(self.dataset.filtered_spikes_with_stimulus_path(self.stimulus_id) + '/*.pkl')]
        
        # return [Path(f).stem for f in glob.glob(self.dataset.filterd_spike_path + '/*.pkl')]

    def get_units_in_recording(self, rec_id, unit_id=None):
        '''
        :param unit_id, additional info about the unit, useful if filtering over best parent set data
        then it will only account recordings which are in best pars for unit_id 
        '''
        units = list(self.data[self.data.recording == rec_id].id.unique())
        units.remove(unit_id) ## remove the Y 
        return units


    def get_DI_path(self, time):
        '''
        Create(if not exist) DI directory 
        :time the observation window we are considering  
        '''
        path = self.dataset.DI_values_path(time, self.stimulus_id)
        ## Create the directories if not exist  
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        return path


    @property
    def filtered_spikes_path(self):
        return self.dataset.filtered_spikes_with_stimulus_path(self.stimulus_id)


    @property
    def data_dir(self):
        return self.dataset.data_dir 

    def __call__(self):

        search_params = []
        for time in self.times_list:
            for part in self.part_list: 
                units = self.get_list_of_units()
                for unit in units:
                    search_params.append( (unit, time, part) )

        ## Shuffle the list 
        ## This is important to running jobs in parallel working on the same list 
        random.seed(os.urandom(1000))
        random.shuffle(search_params)


        for param in search_params:
            self.get_parents(param)


        print('LOG: search for parents is completed.')




    def sortcells(self, tuplecells):
            
        tuplecells=list(tuplecells)
        tuplecells.sort()

        return tuplecells  


    def compute_DI2(self, Ydict, fit_Y, cells,fitting_times, nobs):
        cells = self.sortcells(cells)
        all_cells = self.get_list_of_units()

        assert [cell in all_cells for cell in cells ]
        Xmat = self.prepare_XPast(cells)

        fit_Y_X = self.model_class(Ydict, Xmat, fitting_times, nobs).fit()
        results = model.ModelComparator(fit_Y, fit_Y_X).compare()
        
        return results



    
    def generate_par_tuples_grd_limited(self, cellY, DI,k):
        """
        multiple parts
        ['all2all', 'groups','layers','layers-groups']    
        :params cellY the central unit/cell we want to model 
        :param DI currently calculated DI values for cellY
        :param k number of parents in a set of parents 

        
        """
        max_k_minus_1 = 100#2 #50 #max number pairs keep
        max_1 = 50#5#20 #max number single parents keep
        
        print('less exhaustive search')
        
        
        top_kminus1 = self.list_top_tuples_fixed_size(cellY,DI,k-1)
        
        top_singlepar = self.list_top_tuples_fixed_size(cellY,DI,1)  
        
        
        #because of non-overlap, initially allow for more groups, then prune
        
        top_kminus1 = [top_kminus1[i][0] for i in range( min(len(top_kminus1), max_k_minus_1*max_1) ) ]        
        
        top_singlepar = [top_singlepar[i][0] for i in range( min(len(top_singlepar), max_k_minus_1*max_1) ) ]    
        
        #try forming all combinations if singlepar not in tuple already, then make sure is sorted
        
        combined = []
        
        for kpars in top_kminus1:
            for singlepar in top_singlepar:
                
                
                if singlepar[0] not in kpars and len(combined) < max_k_minus_1*max_1:
                    tmp = list(kpars)+list(singlepar)
                    tmp = tuple(sorted(tmp))
                    if tmp not in combined:
                        combined.append( tmp   )
        
    
        return combined
        
    def list_top_tuples_fixed_size(self, cellY,DI,k):
        """
        This looks for all tuples of size k, ranks them in decreasing order of DI
        """
        
        list_par_sets = [i for i in DI[cellY].keys() if isinstance(i,tuple) and not isinstance(i[0],int) and len(i)==k]
        
        tmpDI_vals= [(parset, DI[cellY][parset]['DI_X_Y']) for parset in list_par_sets if DI[cellY][parset]]
        
        ordered = sorted(tmpDI_vals, key=lambda tpl: tpl[1], reverse=True)

        return ordered


    def get_active_times(self, times):

        '''
        Cache active times instead of loading it on every call 
        '''
        if times not in self.active_times.keys():
            print('LOG: caching active times..')
            active_times_path = self.dataset.active_times_with_stimulus_path(times, self.stimulus_id) 
            active_time_dict = read_pickle_file(active_times_path)
            
            assert active_time_dict, 'Active time is not calculated yet for stimulus {}'.format(self.stimulus_id) 
            self.active_times[times] = active_time_dict

        return self.active_times[times]

    # def get_recording_past_filter_values(self, recording):

    #     if recording in self.recording_


    def update_past_filter_cache(self, recording, max_time, cellY=None):
        '''
        Updates past_filter incase the recording is changed
        :param cellY additional filtering information 
        '''
        if self.past_filter_cache['recording'] == recording and self.past_filter_cache['cellY'] == cellY:
            ## Reuse cache, don't update
            return 

        del self.past_filter_cache # delete the reference 
        print('LOG: updating past filter cache')
        self.past_filter_cache = {}
        units = sorted(self.get_units_in_recording(recording, cellY))
        
        self.past_filter_cache['cellY'] = cellY
        self.past_filter_cache['units'] = units
        self.past_filter_cache['recording'] = recording
        max_time = max_time or self.config.time_per_trial * self.config.n_trials
        Past = np.zeros((len(units), max_time), dtype=np.uint8)
        for i, unit in enumerate(units):
            unit_spikes = read_pickle_file(self.filtered_spikes_path + os.sep + unit + '.pkl')
            if unit_spikes:                
                # import ipdb; ipdb.set_trace()
                Past[i] = unit_spikes['past_filt'][:, 2]

        self.past_filter_cache['Past'] = Past
        



    def prepare_XPast(self, units):
        '''
        Filters the set of XPast from our past_filter_cache, given the units ids 
        '''
        try:
            indices = [self.past_filter_cache['units'].index(u) for u in units]
            Past = self.past_filter_cache['Past'][indices, :] 
        except:
            import ipdb; ipdb.set_trace()
        return Past.T



    def get_parents(self, tpl):
        '''
        :param tpl 
        :pram config - all configurations of the run
        :param dataset - the dataset we're currently looking at 
        '''

        cellY = tpl[0] #the child unit
        times = tpl[1]
        part = tpl[2] #restrictions on parent sets
        
        maxK = self.config.max_number_parents #maximum number of parents to consider


    
        # nobs = 40*4000  # 40 trials of 4 sec each
        nobs = self.config.time_per_trial * self.config.n_trials
        
        
        # if dirs['save_DIs'] not in os.listdir(dirs['data_dir']):
        #     dir_new = dirs['data_dir'] + os.sep+ dirs['save_DIs']
        #     print('missing directory '+dir_new)
        #     os.makedirs(dir_new, exist_ok=True)
        
        
        ## Get the recording of the unit 
        recording = self.data[self.data.id==cellY].recording.unique()[0]

            
        print('Working on cell '+cellY + ' in rec ' + recording + ' for times ' + str(times) + ' in part ' + str(part) )

        
        #load the active times for this recording session
        
        # with open(dirs['data_dir'] + os.sep+'active_times_'+dataset_name+'.pkl', 'rb') as f:
        #     active_time_dict = pickle.load(f)   
        
        active_time_dict = self.get_active_times(times)
        if active_time_dict is None: 
            print('Skipping; missing active time file')
            return 


        fitting_times = active_time_dict[recording]
        
        
        ## Debugging: Give up if no fitting times 
        if not np.any(fitting_times): 
            print("Skipping due to 0 fitting times")
            return 
        ##
        
        ## update parent past filter cache - this will load the past filter vector values for all cells in the recording 
        self.update_past_filter_cache(recording, nobs, cellY)

        
    #--------------------------------------------------------#
        
        #if haven't already made a file to store values, do so, else load it up
    
        DI_filename = 'DI_dict_'+ cellY +'.pkl'
        DI_file_path = self.get_DI_path(times) + os.sep + DI_filename
        
        DI = read_pickle_file(DI_file_path)

        if DI:
            for k in range(1,maxK+1):
                if (k,part) not in DI[cellY].keys():
                    DI[cellY][(k,part)] = {}
                    
        else: #save empty
            DI = {}
            DI[cellY] = {}
            for k in range(1,maxK+1):
                if (k,part) not in DI[cellY].keys():
                    DI[cellY][(k,part)] = {}
            pickle_object(DI, DI_file_path)
            

        # Fitting the model with no parents - and reuse it in the comparisions 
        Ydict = read_pickle_file(self.filtered_spikes_path + os.sep + cellY + '.pkl')
        fit_Y = self.model_class(Ydict, None, fitting_times, nobs).fit()
    #--------------------------------------------------------#
        
        #generate an initial list of candidate parents

        # all_cells = self.get_list_of_units() ## I should replace all "cell"s to "unit"s one day :)
        all_cells = self.get_units_in_recording(recording, unit_id=cellY)
        

        #make a separate copy
        pot_pars = list(all_cells) ## pot_pars - potential parents 
        
        if not pot_pars:
            print('Skipping, no potential parents')
            return 

        del all_cells #so don't accidentally reuse
        DI[cellY][(1,part)]['pot_pars'] = pot_pars

        
    #--------------------------------------------------------#    
        
    #     Now call for (pairwise) DI.  I(X \to Y)
        

        if 'completed' not in DI[cellY][(1,part)].keys() or  not DI[cellY][(1,part)]['completed']:
                
            updated = False #track if changed since last save
            for cellX in pot_pars:           
        
                if tuple([cellX]) not in DI[cellY].keys():
                    # results = self.compute_DI2_alt(cellY, tuple([cellX]), fitting_times,nobs)
                    results = self.compute_DI2(Ydict, fit_Y, (cellX, ),fitting_times, nobs)
                    
                    DI[cellY][tuple([cellX])] = results
                    updated = True
        
            #checkpoint
                if pot_pars.index(cellX) % 100==0 and updated:
                    pickle_object(DI, DI_file_path)
                    updated = False
                    
                    
                if pot_pars.index(cellX) % 100==0:
                    print('Cell [k=1]   '+ str(pot_pars.index(cellX)) +' out of '+ str(len(pot_pars)) )
        
        
            DI[cellY][(1,part)]['completed'] = True
        #    do a final save, only if did new calculations
            if updated:
                pickle_object(DI, DI_file_path) 
                updated = False
        print('Cell [k=1]   complete')
        
    #----------------------  k>1 -----------------------------#    
        
        for k in range(2,maxK+1):
            
            
            if 'completed' in DI[cellY][(k,part)] and DI[cellY][(k,part)]['completed']:
                print('Cell [k='+str(k)+']   already completed')
                continue
            
            updated = False
            # Now call for (all pairs) DI.  I(X \to Y)
            pot_pars = self.generate_par_tuples_grd_limited(cellY, DI,k) 
            
            
            if len(pot_pars) > 10000:
                print('\n\n\t Too many pot_pars -- trimming to 10000')
                random.shuffle(pot_pars)
                pot_pars = pot_pars[:10000]
            #first store these  
            
            
            DI[cellY][(k,part)]['pot_pars'] = pot_pars
            
            
            for pars in pot_pars:
                
                if pars not in DI[cellY].keys():
            
                    # results = self.compute_DI2_alt(cellY, pars, fitting_times,nobs )
                    results = self.compute_DI2(Ydict, fit_Y, pars, fitting_times, nobs)
                    
                    DI[cellY][pars] = results
                    updated = True

        #checkpoint
                if pot_pars.index(pars) % 100==0 and updated:
                    pickle_object(DI, DI_file_path)
                    updated = False
                            
            DI[cellY][(k,part)]['completed'] = True
            #    do a final save
            
            pickle_object(DI, DI_file_path)

            print('Cell [k='+str(k)+']   complete') 

        return




def main(config):
    
    for dataset in config.datasets:
    # for dataset, prefiltered_dataset in zip(datasets, prefiltered_datasets):
        
        ## Load the data 
        
        print('Loading dataset {}'.format(dataset))
        data = pandas.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or list(data.stimn.unique())

        # random.shuffle(stimuli)
        for stimulus_id in stimuli:
            SearchParentSet(config, dataset, data, stimulus_id=int(stimulus_id))()
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for parent sets')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml')
    parser.add_argument('--model', action='store', type=str, default='mdl', choices=['mdl', 'tts']) ## mdl vs tts 

    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config, model_class=args.model)
    
    main(conf)
