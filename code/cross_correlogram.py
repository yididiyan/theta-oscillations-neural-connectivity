import argparse
from multiprocessing import reduction

import utils
from config import Configuration
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import re 
import os 
import glob 
import pickle 
from pathlib import Path

"""
Deprecated 
"""


class CrossCorrelogram:


    def __init__(self, config, dataset, 
                data, 
                di_dir, 
                stimulus_id = None,
                recordings=None,
                rank_by_area=False # filter units by area for plotting (ranking by area)
                ) -> None:
        self.config = config 
        self.dataset = dataset 
        self.data = data
        self.di_dir = di_dir
        self.stimulus_id = stimulus_id
        self.recordings = recordings
        self.rank_by_area = rank_by_area
        print(f'Ranking by area {self.rank_by_area}')

        self.W = 50 



        # self.data['times_per_trial'] = self.data.times / (self.data.trial + 1. )
        self.data['times_per_trial'] = self.data.times % self.config.time_per_trial 

        # slice data by window 
        self.trials = sorted(self.data.trial.unique())
        # self.data = self.data[ (self.data.times_per_trial > window[0]) & ( self.data.times_per_trial < window[1] )]



    def make_binary_vec(self, unit_id):
        """
        unit_id  - unit id 
        """

        vecs = []
        print('Building binary vector... ')
        assert self.data[self.data.id == unit_id] 
        for t in self.trials: # all trials
            
            train = self.data[(self.data.trial == t) & 
                                        (self.data.id  == unit_id) & 
                                        (self.data.times_per_trial > self.config.observation_window[0] )  & 
                                        (self.data.times_per_trial < self.config.observation_window[1])].times_per_trial.astype(int).tolist()
            
            vec = np.zeros(self.config.time_per_trial)
            vec[train] = 1
            vecs.append(vec)
        
        return vecs

    def prep_spikes(self, unit_id):
        """
        unit_id  - unit id 
        """
        W = self.W 
        vecs = []
        for t in self.trials: # all trials 
            
            train = self.data[(self.data.trial == t) & 
                                        (self.data.id  == unit_id) & 
                                        (self.data.times_per_trial > self.config.observation_window[0]+ W  )  & 
                                        (self.data.times_per_trial < self.config.observation_window[1]- W )].times_per_trial.astype(int).tolist()
            

            vecs.append(train)
        
        return vecs


    def __call__(self):
        import os 
        import re 
        # create a cross-correlogram directory
        cc_dir = self.di_dir + '../cc'
        os.makedirs(cc_dir, exist_ok=True)

        files = glob.glob(self.di_dir + '/*.pkl')
        W = self.W

        for f in files:
            # X - units in parentset , Y - the child unit 
            data = pickle.load(open(f, 'rb')) 

            # unit ids may contain some prefix 00ET, remove them.  
            unitY = re.sub('DI_dict_\d*', '', Path(f).stem)
            unitXs = data['selected_parents']
            unitXs = [re.sub('\d*ET', 'ET', u) for u in unitXs ] 
            
            areaXs = [ u.split('_')[-2] for u in unitXs ]
            areas = list(set(areaXs))


            # check recordings 
            if self.recordings is not None: 
                if not [r for r in self.recordings if r in f]:
                    # not included in the recordings filter 
                    continue 
    

            if data['DI_Y_X_all'] < 0.:
                continue # skip analysis if there parents don't inform much 

            reduction_percentage =  (data['HY_all'] - data['HY_X_all']) / data['HY_all'] * 100
            print('Entropy reduction in percentage ', reduction_percentage)
            
            if reduction_percentage < 10:
                continue  # skip low entropy reduction   

            # ref unit is parent unit -- unitXs 
            # for every parent spike, we look at the spikes of the child unit [t-W, t+ W] 

            # prep ref spikes 

            if not len(unitXs):
                continue # skip analysis 


            parent_spikes = [ self.prep_spikes(unitX) for unitX in unitXs ]  
            child_spikes = self.make_binary_vec(unitY)


            assert len(parent_spikes) == len(unitXs)
            assert len(child_spikes) == len(self.trials)

            correlation = np.zeros((len(unitXs), 2 * W + 1 )) # for each parent 
            spike_counts = [1e-10] * len(unitXs) # to avoid zero division
             
            # for each trial
            for t in range(len(self.trials)):
                # for each parent 
                for j in range(len(unitXs)):
                    for s in parent_spikes[j][t]:
                        # for each parent spike time 
                        correlation[j] += child_spikes[t][s-W: s+W+1]
                        spike_counts[j] += 1 

            
            
            # normalize correlation by # parent spikes 
            correlation = correlation / np.array(spike_counts)[...,None]


            # compute base value for correction 
            correlation_base = np.hstack(
                                    (correlation[:, :20], correlation[:,-20:]) 
                                ).mean(-1, keepdims=True)

            
            # adjust correlation by base value  
            correlation_adjusted = correlation - correlation_base 
            threshold = 3 * correlation_adjusted.std(-1) # threshold - 3 std bounding the mean

            # look at peaks and troughs of the bins  -- [0 - 10] -- 11 bins in total 
            extremes = np.abs(
                    correlation_adjusted[:, W: W + 11] / (threshold[..., None]  + 1e-10)
                ).max(-1)



            
            if not self.rank_by_area:
                # TOP three units  from all areas 
                top_three_indices = np.argsort(extremes)[::-1][:3]

                fig, axes = plt.subplots(3, figsize=(8, 15))


                for ax, idx in zip(axes, top_three_indices):
                    ax.bar(range(-W, 0), correlation_adjusted[idx][:W], align='center', alpha=0.25, color='blue', width=1.0,linewidth=0)
                    ax.bar([0], correlation_adjusted[idx][W], align='center', alpha=1.0, color='black', width=1.0,linewidth=0)
                    ax.bar(range(1, W+1), correlation_adjusted[idx][-W:], align='center', alpha=0.25, color='red', width=1.0,linewidth=0)


                    # plt.ylim(top=0.25, bottom=-0.25) 
                    ax.set_xlim(right=W, left=-W)
                    ax.axhline(-threshold[idx],linestyle='--', color='gray')
                    ax.axhline(threshold[idx], linestyle='--', color='gray')
                    ax.set_title(f'{unitXs[idx]} -> {unitY}')

                    ax.set_xlabel('Time [ms]')
                    ax.set_ylabel('Spike Trans Prob')
                    
                fig.tight_layout()
                fig.savefig(f'{cc_dir}/{unitY}.png')

            else: 

                for a in areas: 

                    # select members  
                    unit_idx = [ i for i, u  in enumerate(unitXs) if f'_{a}_' in u ]
                    
                    # filter top indices by area 
                    top_three_indices = [ idx for idx in  np.argsort(extremes)[::-1] if idx in unit_idx ][:3] # top 3 

                    fig, axes = plt.subplots(3, figsize=(8, 15))


                    for ax, idx in zip(axes, top_three_indices):
                        ax.bar(range(-W, 0), correlation_adjusted[idx][:W], align='center', alpha=0.25, color='blue', width=1.0,linewidth=0)
                        ax.bar([0], correlation_adjusted[idx][W], align='center', alpha=1.0, color='black', width=1.0,linewidth=0)
                        ax.bar(range(1, W+1), correlation_adjusted[idx][-W:], align='center', alpha=0.25, color='red', width=1.0,linewidth=0)


                        # plt.ylim(top=0.25, bottom=-0.25) 
                        ax.set_xlim(right=W, left=-W)
                        ax.axhline(-threshold[idx],linestyle='--', color='gray')
                        ax.axhline(threshold[idx], linestyle='--', color='gray')
                        ax.set_title(f'{unitXs[idx]} -> {unitY}')

                        ax.set_xlabel('Time [ms]')
                        ax.set_ylabel('Spike Trans Prob')
                        
                    fig.tight_layout()
                    fig.savefig(f'{cc_dir}/{a}_{unitY}.png')









def main(config, di_dir, recordings=None, rank_by_area=False):
    for dataset in config.datasets:

        print('Working with {}'.format(dataset))
        data = pd.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or list(data.stimn.unique())

        for stimulus_id in stimuli:
            CrossCorrelogram(config, dataset, data, di_dir, stimulus_id=int(stimulus_id), recordings=recordings, rank_by_area=rank_by_area)()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for parent sets')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml')
    parser.add_argument('--all-areas', nargs='+', help='List of all areas', required=False, type=str, default=None)
    parser.add_argument('--recordings', nargs='+', help='List of recordings', required=False, type=str, default=None)
    parser.add_argument('--di-dir', help='DI directory', required=False, type=str, default=None)

    parser.add_argument('--rank', help='Rank units by area -- for plotting CC', required=False, type=bool, default=None)
    
    

    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config)
    main(conf, args.di_dir,  recordings=args.recordings, rank_by_area=args.rank)
