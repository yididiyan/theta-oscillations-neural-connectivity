import argparse
from functools import lru_cache
from multiprocessing import reduction

import utils
from config import Configuration
import pandas as pd 
import numpy as np 
import matplotlib

import matplotlib.pyplot as plt 

import re 
import os 
import glob 
import pickle 
from pathlib import Path

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


""""

We discuss the procedure to compute the cross correlgoram. Consider two simultaneously recorded units namely X and Y. To compute the cross-correlogram . First, we iterate over the spikes times of unit A. For each spike time of unit A $t$, we look for spikes in the time window of size W before and after the spike.
We then compute their differences with $t$. We rounded down the values to the nearest integer and aggregate the results into 2 * width . We sum these binned differences across multiple trials and normalize them with the total number of spikes in unitA.
We present the results in a histogram plot. 

"""

class CrossCorrelogram:


    def __init__(self, config, dataset, 
                data, 
                di_dir, 
                stimulus_id = None,
                recordings=None,
                rank_by_area=False, # filter units by area for plotting (ranking by area)
                pdf=False,
                entropy_reduction_threshold=0. # minmum reduction percentage (DI perrcentage) to compute correlation 
                ) -> None:
        self.config = config 
        self.dataset = dataset 
        self.data = data
        self.di_dir = di_dir
        self.stimulus_id = stimulus_id
        self.recordings = recordings
        self.rank_by_area = rank_by_area
        self.pdf=pdf 
        print(f'Ranking by area {self.rank_by_area}')

        self.W = 50 

        # filter data by stimulus id if exist 
        if self.stimulus_id is not None:
            self.data = self.data[self.data.stimn == float(self.stimulus_id)]
            # reset index 
            self.data = self.data.reset_index(drop=True)
        
        assert len(self.data) != 0, f'empty dataframe for stimulus id {self.stimulus_id}'

        # self.data['times_per_trial'] = self.data.times / (self.data.trial + 1. )
        self.data['times_per_trial'] = self.data.times % self.config.time_per_trial 

        # slice data by window 
        self.trials = sorted(self.data.trial.unique())
        # self.data = self.data[ (self.data.times_per_trial > window[0]) & ( self.data.times_per_trial < window[1] )]


        self.entropy_reduction_threshold = entropy_reduction_threshold
    


    def make_vec(self,unit_id):
        vecs = []
        print('Building continuous time vector... ')
        for t in self.trials: # all trials
            
            train = self.data[(self.data.trial == t) & 
                                        (self.data.id  == unit_id) & 
                                        (self.data.times_per_trial > self.config.observation_window[0] )  & 
                                        (self.data.times_per_trial < self.config.observation_window[1])].times_per_trial.tolist()
            
            vecs.append(train)
        
        return vecs

    @lru_cache(maxsize=2000)
    def pre_spikes_continuous(self, unit_id):
        W = self.W 
        vecs = []
        for t in self.trials: # all trials 
            
            train = self.data[(self.data.trial == t) & 
                                        (self.data.id  == unit_id) & 
                                        (self.data.times_per_trial > self.config.observation_window[0]+ W  )  & 
                                        (self.data.times_per_trial < self.config.observation_window[1]- W )].times_per_trial.tolist()
            

            vecs.append(train)
        
        return vecs

    @lru_cache(maxsize=2000)
    def units(self, recording):
        return self.data[(self.data.recording == recording) ].id.unique().tolist()


    def compute_correlations(self, unitXs, unitY):
        print(f'Working on {unitY} ...')
        W = self.W 
        parent_spikes = [ self.pre_spikes_continuous(unitX) for unitX in unitXs ]  
        child_spikes = self.make_vec(unitY)

        print(self.pre_spikes_continuous.cache_info(), ' cache info')

        assert len(parent_spikes) == len(unitXs)
        assert len(child_spikes) == len(self.trials)

        # correlation = np.zeros((len(unitXs), 2 * W + 1 )) # for each parent 
        correlation = np.zeros((len(unitXs), 2 * W )) # for each parent 
        spike_counts = [1e-10] * len(unitXs) # to avoid zero division
        
        # for each trial
        for t in range(len(self.trials)):
            # for each parent 
            child_spikes_t = np.array(child_spikes[t])
            for j in range(len(unitXs)):
                for s in parent_spikes[j][t]:
                    # for each parent spike time 
                    diff = child_spikes_t[(child_spikes_t > s-W) & (child_spikes_t < s+W)] - s

                    # remove exact conincidences 
                    diff = diff[diff  != 0.] 
                    # diff of 0.5 -- falls into the 'W'th bin (positive lag) 
                    # diff of -0.5 --falls into the W-1 th bin (negative lag)
                    diff = np.floor(diff).astype(int) 

                    assert np.all(np.abs(diff) <= W ) 
                    diff = diff + W  # shift diffs by W to match #bins 
                    for d in diff: 
                        correlation[j][d] += 1 

                    spike_counts[j] += 1 # computed correlation with one parent spike "s"


                    
        
        # normalize correlation by # parent spikes 
        correlation_raw = correlation
        correlation = correlation / np.array(spike_counts)[...,None]


        # compute base value for correction 
        correlation_base = np.hstack(
                                (correlation[:, :20], correlation[:,-20:]) 
                            ).mean(-1, keepdims=True)

        
        # adjust correlation by base value  
        correlation_adjusted = correlation - correlation_base 

        # look at peaks and troughs of the bins  -- [1 - 10] -- 10 bins in total 
        # extremes = np.abs(
        #         correlation_adjusted[:, W: W + 10] / (threshold[..., None]  + 1e-10)
        #     ).max(-1)

        
        # pickle the results for later 
        result = {
            'spike_counts': spike_counts, # spike counts of parents across trials 
            'correlation_raw': correlation_raw, 
            'correlation_adjusted': correlation_adjusted, 
            'correlation': correlation, # normalized by parent spikes  
            'unitY': unitY,
            'unitXs': unitXs,
            'W': self.W ,
            # entropy 
        }

        return result 

    def __call__(self, all_units=True):
        """
        all_units - use all pairs of units -- not just ones selected by the DI lasso method 
        """

        # create a cross-correlogram directory
        cc_dir = self.di_dir + '../cc'
        os.makedirs(cc_dir, exist_ok=True)
        print(f'Saving results under {cc_dir}')
        
        W = self.W

        if all_units:
            units = self.data.id.unique().tolist()
            for unitY in units:
                # find the recording of unitY 
                rec = self.data[self.data.id == unitY].recording.unique()[0]
                # find the list of units in that recording 
                # unitXs = self.data[(self.data.recording == rec) & (self.data.id != unitY) ].id.unique().tolist()
                unitXs = self.units(rec)
                unitXs = [ u for u in unitXs if u != unitY]

                areaXs = [ u.split('_')[-2] for u in unitXs ]


                # check recordings 
                if self.recordings is not None: 
                    if not [r for r in self.recordings if r in unitY]:
                        # not included in the recordings filter 
                        continue 

                

                result = self.compute_correlations(unitXs, unitY)

                pickle.dump(result, open(f'{cc_dir}/correlation-{unitY}.pkl', 'wb'))


        else:
            # DEPRECATED 
            # PREVIOUS code that restricts unit-to-unit pairs to those pairs selected for the 
            files = glob.glob(self.di_dir + '/*.pkl')
            units = [re.sub('DI_dict_\d*', '', Path(f).stem) for f in files ]
        

            for i, unitY in enumerate(units):
            
                # X - units in parentset , Y - the child unit 
                data = pickle.load(open(files[i], 'rb')) 

                unitXs = data['selected_parents']            
                unitXs = [re.sub('\d*ET', 'ET', u) for u in unitXs ] 
                
                areaXs = [ u.split('_')[-2] for u in unitXs ]
                areas = list(set(areaXs))

                 # check recordings 
                if self.recordings is not None: 
                    if not [r for r in self.recordings if r in unitY]:
                        # not included in the recordings filter 
                        continue 
                if data['DI_Y_X_all'] < 0.:
                    continue # skip analysis if there parents don't inform much 

                reduction_percentage =  (data['HY_all'] - data['HY_X_all']) / data['HY_all'] * 100
                print('Entropy reduction in percentage ', reduction_percentage)
                
                if reduction_percentage < self.entropy_reduction_threshold:
                    continue  # skip low entropy reduction   

                # ref unit is parent unit -- unitXs 
                # for every parent spike, we look at the spikes of the child unit [t-W, t+ W] 

                # prep ref spikes 

                if not len(unitXs):
                    continue # skip analysis



                result = self.compute_correlations(unitXs, unitY)


                if not all_units and data:
                    result['HY_all'] = data['HY_all']
                    result['HY_X_all'] = data['HY_X_all']

                pickle.dump(result, open(f'{cc_dir}/correlation-{unitY}.pkl', 'wb'))


        return 


def main(config, di_dir, recordings=None, rank_by_area=False, pdf=False):
    for dataset in config.datasets:

        print('Working with {}'.format(dataset))
        data = pd.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or list(data.stimn.unique())

        for stimulus_id in stimuli:
            CrossCorrelogram(config, 
            dataset, 
            data, 
            di_dir, 
            stimulus_id=int(stimulus_id), 
            recordings=recordings, 
            rank_by_area=rank_by_area, 
            pdf=pdf)()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for parent sets')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml')
    parser.add_argument('--all-areas', nargs='+', help='List of all areas', required=False, type=str, default=None)
    parser.add_argument('--recordings', nargs='+', help='List of recordings', required=False, type=str, default=None)
    parser.add_argument('--di-dir', help='DI directory', required=False, type=str, default=None)

    parser.add_argument('--rank', help='Rank units by area -- for plotting CC', required=False, type=bool, default=None)
    parser.add_argument('--pdf', help='Plot to PDF file', required=False, type=bool, default=False)
    
    

    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config)
    main(conf, args.di_dir,  recordings=args.recordings, rank_by_area=args.rank, pdf=args.pdf)


#  python cross_correlogram.py --config   /media/yido/additional/research/PurdueZimmerman/cross_correlogram/output__stimuli_0/runner_config_V1RSC_spks_CQuinn_post.feather.yml  --all-areas 'V1 RSC'  --di-dir ../../V1RSC/output__stimuli_0/DI_values/V1RSC_spks_CQuinn_post/0/500_1500/lasso_0.5/
#  python cross_correlogram.py --config   /media/yido/additional/research/PurdueZimmerman/cross_correlogram/output__stimuli_0/runner_config_V1LP_spks_CQuinn_post.feather.yml  --di-dir ../../V1LP/output__stimuli_0/DI_values/V1LP_spks_CQuinn_post/0/500_1500/lasso_0.5/ --rank True