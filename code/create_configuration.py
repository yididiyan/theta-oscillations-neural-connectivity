'''
This module slices the dataset with given configuration to prepare a new dataset
'''
import os 
import argparse
import pandas as pd
import json 
import yaml


from pathlib import Path

class CreateConfiguration(object): 
    def __init__(self, options) -> None:


        print('Loading dataset...')
        self.data = pd.read_feather(options['dataset'])
        
        self.output_dir = '{}_stimuli_{}'.format(options['output_dir'], '_'.join(map(str, options['stimuli'])))
        


        # make output dir if not exist 
        os.makedirs(self.output_dir, exist_ok=True)

        # self.output_filename= self.output_dir + os.sep + Path(options['dataset']).name
        self.dataset_name = Path(options['dataset']).name
        self.dataset_path = options['dataset']
        self.stimuli = options['stimuli']
        self.n_recordings = options['n_recordings'] ## limit the number of mice 
        self.observation_window = options['observation_window']
        self.time_per_trial = options['time_per_trial']



        # Save the settings in the output dir 
        print('Saving configurations')
        with open(self.output_dir + '/settings.json', 'w') as f:
            json.dump(options, f)

        self.data = self.data[self.data['stimn'].isin(self.stimuli) ] 
        

        ## Create a submission script 
        self.create_submission_script(options)



        ## Assertions 
        assert os.path.exists(self.output_dir + '/runner_config_{}.yml'.format(self.dataset_name)), "Runner configuration is not properly created"

    def create_submission_script(self, options):
        config = None 


        with open('../configs/default_config.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        config = config or {
            'observation_window': '500, 1500', 
            'max_number_parents': 3 , 
            'markov_order': 10,
        }
        config['n_trials'] = len(self.data.trial.unique())
        config['datasets'] = [os.path.abspath(options['dataset'])]
        config['data_dir'] = os.path.abspath(self.output_dir)
        config['stimuli'] = self.stimuli
        config['n_recordings'] = self.n_recordings
        config['observation_window'] = self.observation_window
        config['time_per_trial'] = self.time_per_trial

        print('Configuration {}'.format(config))
        # update configuration and save it 
        with open(self.output_dir + '/runner_config_{}.yml'.format(self.dataset_name), 'w') as f:
            
            yaml.dump(config, f)
    





if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Configure a runner for specific stimulus')
    parser.add_argument('--stimuli', nargs='+', help='List of stimuli ids, zero based indexing', required=True, type=int)
    parser.add_argument('--n_recordings', help='Number of recordings to consider', default=10000, type=int)
    parser.add_argument('--trial_length', help='Time per trial (in ms.)', default=2000, type=int)
    parser.add_argument('--dataset', help='File path of the feather dataset you wish to slice',  required=True)
    parser.add_argument('--output_dir', help='Results output directory', required=True)
    parser.add_argument('--window', help='Observation window', default='500, 1500')



    ## make sure to subtruct 1 from stimuli indices 

    args = parser.parse_args()
    # parse args 
    options = {
        'stimuli': args.stimuli,
        'dataset': args.dataset,
        'n_recordings': args.n_recordings, 
        'output_dir': args.output_dir,
        'observation_window': args.window,
        'time_per_trial': args.trial_length
    }

    CreateConfiguration(options)