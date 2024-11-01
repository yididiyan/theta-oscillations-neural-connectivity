import argparse
from multiprocessing.spawn import prepare
import random
import os
import glob 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import pickle_object
from scipy.sparse import csr_matrix


# sys.path.append('../../code/')
# sys.path.append('../../analysis/')

import utils
from config import Configuration



TEST_SIZE = 0.2
STIMULI = list(range(0, 25, 1))
# STIMULI = [8,7,6,5,4,3]

AREAS = ['AL', 'LM', 'V1']


## grid search for inverse lambdas 
# Cs = np.logspace(np.log10(0.03), np.log10(1.7), 20)
Cs = 40

class LassoParentSelection():

    def __init__(self, config, dataset, 
                data, stimulus_id = None,
                penalty='l1', solver='liblinear', # convergence issues using saga
                cv = 5, 
                Cs = Cs,
                using_all_data=False,
                area_filter = None,
                calculate_rates=False,
                test_size=0.5, # test size to use
                all_areas=None,
                ) -> None:
        self.config = config
        self.calculate_rates = calculate_rates
        self.dataset = dataset 
        self.data = data 
        self.stimulus_id = stimulus_id
        
        # lasso options 
        self.penalty= penalty 
        self.solver = solver
        self.cv = cv ## cross validation k value 
        self.Cs = Cs
        
        self.all_areas = all_areas or AREAS
        self.area_filter = area_filter
        
        if self.area_filter:
            print(f'Using area filter {self.area_filter}')


        self.active_times  = {}
        self.times  = '_'.join(map(str, self.config.observation_window)) 
        self.using_all_data = using_all_data
        self.test_size = test_size
        

    def get_active_times(self, stimulus_id):

        '''
        Cache active times instead of loading it on every call 
        stimulus_id - value to override configurations stimulus_id 
        '''
        key = f'{self.times}_{str(stimulus_id)}'
        if key not in self.active_times.keys():
            print('LOG: caching active times..')
            active_times_path = self.dataset.active_times_with_stimulus_path(self.times, stimulus_id) 
            active_times_path = active_times_path.replace(f'output_stimuli_{str(self.stimulus_id)}' , f'output_stimuli_{str(stimulus_id)}') # configuration is only for a single stimulus, thus needs update 
            
            active_time_dict = utils.read_pickle_file(active_times_path)
            
            assert active_time_dict, 'Active time is not calculated yet for stimulus {}'.format(stimulus_id) 
            self.active_times[key] = active_time_dict

        return self.active_times[key]

    def get_list_of_units(self):

        ## Get list of all units 
        if self.stimulus_id is not None:
            return self.data[self.data.stimn == self.stimulus_id].id.unique()
        
        return self.data.id.unique()

    def get_list_of_recordings(self):
        ## Get list of all units 
        if self.stimulus_id is not None:
            return self.data[self.data.stimn == self.stimulus_id].recording.unique()
        
        return self.data.recording.unique()


    @property
    def filtered_spikes_path(self):
        if self.stimulus_id is not None:
            return self.dataset.filtered_spikes_with_stimulus_path(self.stimulus_id)

        return self.dataset.filtered_spikes_path


    
    @property
    def data_dir(self):
        return self.dataset.data_dir 



    def prepare_data(self, Yunit, recording_id, standardize=False):
        '''
        :param Yunit -unit id 
        :param recording_id - recording_id 
        :param standardize - boolean whether to standardize or not 
        '''


        def prepare_data_(stimulus_id, Xunits):
            '''
            prepares data for each stimulus
            '''
            filtered_spikes_path = self.filtered_spikes_path.replace(f'output_stimuli_{str(self.stimulus_id)}' , f'output_stimuli_{str(stimulus_id)}').replace(f'_stimuli_{str(self.stimulus_id)}' , f'_stimuli_{str(stimulus_id)}') ## Switch to another stimulus 
            filtered_spikes_path = filtered_spikes_path.replace(f'/{self.stimulus_id}', f'/{stimulus_id}')
            print(f'Preping data from {stimulus_id}.. reading filtered_spikes from {filtered_spikes_path} ')
            
            Ydict = utils.read_pickle_file(filtered_spikes_path + os.sep + Yunit + '.pkl')
            if not Ydict:
                print('Warning: Y data not found, skipping.. ')
                return None


            Yvec = Ydict['spike_vec']
            Past = Ydict['past_filt'][:,:2] #first variable is for past two ms, second is for previous 8 ms

            ## grab all units under the same recordings 
            files = glob.glob(filtered_spikes_path + '/' + recording_id +  '*.pkl')
            unit_files = {Path(f).stem: f for f in files if Path(f).stem != Yunit}
            

            
            XPast  = [ utils.read_pickle_file(unit_files[u])['past_filt'][:,-1] if unit_files.get(u, None) else np.zeros(Yvec.shape[0]) for u in Xunits]

            XPast = np.array(XPast).T

            ## Filter by fitting times 
            active_times = self.get_active_times(stimulus_id=stimulus_id)

            fitting_times = active_times[recording_id]

            

            Past = np.concatenate((Past, XPast), axis=1)

            Yvec= Yvec[fitting_times[:,0]==1]
            Past = Past[fitting_times[:,0]==1,:]
            Yvec = Yvec.squeeze()


            if Yvec.sum() < 5: 
                # don't bother splitting, very few spikes 
                print(f'too few spikes for {stimulus_id}')
                return None


            ## Train test split 
            Past_train, Past_test, Yvec_train, Yvec_test = train_test_split(Past, Yvec, test_size=self.test_size, shuffle=True, stratify=Yvec, random_state=1000)

            
            return Past_train, Yvec_train, Past_test, Yvec_test 
            

        Xunit_ids = [ Path(x).stem for x in glob.glob(self.filtered_spikes_path + os.sep + recording_id + '*.pkl')]
        Xunit_ids = sorted(Xunit_ids)

        # check if matches area_filter 
        if self.area_filter:
            Xunit_areas = set([ u.split('_')[-2] for u in Xunit_ids ])
            if not sorted(self.area_filter) == sorted(Xunit_areas):
                print(f'Skipped {Yunit} -- area filter in action')
                return None, None, None, None, Xunit_ids


       

        
        Past_train, Yvec_train, Past_test, Yvec_test =  {}, {}, {}, {}
        
        for i in STIMULI:

            data = prepare_data_(i, Xunit_ids) 
            if data == None:
                continue
            
            Past_train[i] = data[0]
            Yvec_train[i] = data[1]
            Past_test[i] = data[2]
            Yvec_test[i] = data[3]

        if self.stimulus_id not in Yvec_train.keys():
            # no data for the current stimulus 
            print(f'Skipping: too few spikes for stimulus {self.stimulus_id }')
            return None, None, None, None, Xunit_ids
           

        Past_train, Yvec_train = np.concatenate(list(Past_train.values())), np.concatenate(list(Yvec_train.values()))
        

        # sparsify matrix 
        # Yvec, Past = Yvec[:10000], Past[:10000]
        Past_train = csr_matrix(Past_train)
        


        if Yvec_train.sum() < 5:
            print('Don\'t bother; too few spikes ')
            return None, None, None, None, Xunit_ids
       
        
        return Past_train, Past_test, Yvec_train, Yvec_test, Xunit_ids

    def fit_model(self, Past_train, Yvec_train, n_jobs=8, C=1.0):

        clf = LogisticRegression(penalty=self.penalty, solver=self.solver, C=C)

        clf.fit(Past_train, Yvec_train)

        return clf, []

    def fit_cv_model(self, Yunit_id, Xunit_ids, Past_train, Yvec_train, n_jobs=8, Cs=None):
        
        Cs = Cs or self.Cs
        clf = LogisticRegressionCV(Cs=Cs, penalty=self.penalty, solver=self.solver, cv=self.cv, scoring='neg_log_loss', n_jobs=n_jobs)
                    
        clf.fit(Past_train, Yvec_train)

        print('Best regularizer: C={}'.format(clf.C_))

        print('Best Parents for unit {}'.format(Yunit_id))
        
        if Xunit_ids:
            assert len(Xunit_ids)  + 2 == len(clf.coef_[0]), 'mismatch in attributes count'


        best_set = []
        for i, u in enumerate(Xunit_ids):
            if clf.coef_[0][i + 2] != 0.:
                best_set.append(u)
        
        # print('Selected {} parents'.format(best_set))


        return clf, best_set




        
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





    def __call__(self):
        units = self.get_list_of_units()
        random.shuffle(units)


        rates = {}

        for Yunit_id in units:

            # check if already done 
            DI_filename = 'DI_dict_'+ Yunit_id +'.pkl'
            DI_file_path = self.get_DI_path(self.times) + os.sep + f'lasso_{self.test_size}' + os.sep + DI_filename
            
            if not self.calculate_rates and os.path.exists(DI_file_path):
                continue

            # prepare data 
            recording_id = '_'.join(Yunit_id.split('_')[:3])
            Past_train, Past_test, Yvec_train, Yvec_test, Xunit_ids = self.prepare_data(Yunit_id, recording_id)

            if Yvec_train is None:
                continue
            
            if self.calculate_rates:
                rates[Yunit_id] = Yvec_train.mean() * 1000 # firing times per second 
                continue

            # fit model - without parents - use lasso regularizer ?? - necessary 
            orphan_model, _ = self.fit_model(Past_train[:, :2], Yvec_train)
            # fit model - with all parents 
            full_model, best_set = self.fit_cv_model(Yunit_id, Xunit_ids, Past_train, Yvec_train, n_jobs=8)
            


            ## decompose best_set by areas and fit model with each subset, separate set by sign of coefficient 
            full_model_coefs = full_model.coef_.squeeze()
            area_units, subset_indices = {a:[] for a in self.all_areas} , {a:{'+ve': [], '-ve': [], 'all':[]} for a in self.all_areas }
            for xu in Xunit_ids: area_units[xu.split('_')[-2]].append(xu)

            # categorize units by sign of coefficient
            for a in area_units.keys(): 
                for u in area_units[a]: ## for unit in the area 
                    idx = Xunit_ids.index(u) + 2 # "2" to account for Y's own feature 
                    if full_model_coefs[idx] > 0.: 
                        subset_indices[a]['+ve'].append(idx)
                    elif full_model_coefs[idx] < 0.:
                        subset_indices[a]['-ve'].append(idx)
                    subset_indices[a]['all'].append(idx)

            # calculate entropy HY, HY_X 

            # prepared data to save
            HY, HY_all = {}, None
            HY_X, HY_X_all = {}, None
            DI_Y_X, DI_Y_X_all = {}, None
            ATE = {} # average treatment effect 

            Past_test_all, Yvec_test_all = np.concatenate(list(Past_test.values())), np.concatenate(list(Yvec_test.values()))
            HY_all = LassoParentSelection.H(orphan_model.predict_proba(Past_test_all[:, :2]), Yvec_test_all)
            HY_X_all = LassoParentSelection.H(full_model.predict_proba(Past_test_all), Yvec_test_all)
            DI_Y_X_all = HY_all - HY_X_all


            # average treament effect -- for all test data
            ## average treatment effect 
            ATE_all = {}
            for a in area_units.keys():
                if area_units[a]:
                    Past_test_with_treatment_0, Past_test_with_treatment_1 = Past_test_all.copy(), Past_test_all.copy()
                    ## Silencing the positive voice, amplifying the negative
                    Past_test_with_treatment_0[:, subset_indices[a]['+ve']] = 0
                    Past_test_with_treatment_0[:, subset_indices[a]['-ve']] = 1
                    
                    ## Amplifying the positive voice, silencing the negative  
                    Past_test_with_treatment_1[:, subset_indices[a]['+ve']] = 1
                    Past_test_with_treatment_1[:, subset_indices[a]['-ve']] = 0

                    ATE_all[a] = (full_model.predict_proba(Past_test_with_treatment_1)[:, 1] - full_model.predict_proba(Past_test_with_treatment_0)[:, 1]).mean()


            for stimulus_id in Past_test.keys():
                
                HY[stimulus_id] = LassoParentSelection.H(orphan_model.predict_proba(Past_test[stimulus_id][: , :2]), Yvec_test[stimulus_id]) 
                HY_X[stimulus_id] = LassoParentSelection.H(full_model.predict_proba(Past_test[stimulus_id]), Yvec_test[stimulus_id]) 
                DI_Y_X[stimulus_id] = HY[stimulus_id] - HY_X[stimulus_id]

                ## average treatment effect -- for just test data under a particular stimulus 
                ATE[stimulus_id] = {}
                for a in area_units.keys():
                    if area_units[a]:
                        Past_test_with_treatment_0, Past_test_with_treatment_1 = Past_test[stimulus_id].copy(), Past_test[stimulus_id].copy()
                        Past_test_with_treatment_0[:, subset_indices[a]['+ve']] = 0
                        Past_test_with_treatment_0[:, subset_indices[a]['-ve']] = 1
                        
                        ## Amplifying the positive voice, silencing the negative  
                        Past_test_with_treatment_1[:, subset_indices[a]['+ve']] = 1
                        Past_test_with_treatment_1[:, subset_indices[a]['-ve']] = 0
                        ATE[stimulus_id][a] = (full_model.predict_proba(Past_test_with_treatment_1)[:, 1] - full_model.predict_proba(Past_test_with_treatment_0)[:, 1]).mean()


           




            print(f'HY {HY[self.stimulus_id]}, HY_X {HY_X[self.stimulus_id]} DI value {DI_Y_X[self.stimulus_id]}')

            results = {
                'HY': HY,
                'HY_X': HY_X,
                'DI_Y_X': DI_Y_X,
                'ATE': ATE,
                'HY_all': HY_all,
                'HY_X_all': HY_X_all,
                'DI_Y_X_all': DI_Y_X_all,
                'ATE_all': ATE_all,
                'selected_parents': best_set,
                'orphan_model': {
                    'intercept': orphan_model.intercept_,
                    'coef': orphan_model.coef_,
                    # 'C': orphan_model.C_,
                    # 'score': np.mean(orphan_model.scores_[1], axis=0)
                },
                'full_model': {
                    'intercept': full_model.intercept_,
                    'coef': full_model.coef_,
                    'C': full_model.C_,
                    'score': np.mean(full_model.scores_[1], axis=0)
                },
                'test_size': self.test_size
            }


            if DI_Y_X[self.stimulus_id] > 0:
                results['area_models'] = {}

                Yunit_area = Yunit_id.split('_')[-2]

                area_models = {}
                area_HY_X = {a: {} for a in self.all_areas}
                area_HY_X_all = {a: None for a in self.all_areas} # with all data combined 
                
                for a in area_units.keys():
                    if area_units[a]:
                        results['area_models'][a] = {}
                        # print(area_units)
                        # include features 0, 1,  Y's own features
                        indices_ = [0, 1] + subset_indices[a]['all']
                        area_models[a], area_best_set = self.fit_cv_model(Yunit_id, area_units[a], Past_train.T[indices_,:].T, Yvec_train, Cs=20)


                        for stimulus_id in Past_test.keys():
                            area_HY_X[a][stimulus_id] = LassoParentSelection.H(area_models[a].predict_proba(Past_test[stimulus_id].T[indices_, :].T), Yvec_test[stimulus_id])
                        
                        area_HY_X_all[a] = LassoParentSelection.H(area_models[a].predict_proba(Past_test_all.T[indices_, :].T), Yvec_test_all)
                        
                        print(f'from area {a} -> {Yunit_area},  HY_X  {area_HY_X[a][self.stimulus_id]} DI {HY[self.stimulus_id] - area_HY_X[a][self.stimulus_id]}')
                        # area_models

                        results['area_models'][a]['HY_X'] = area_HY_X[a]
                        results['area_models'][a]['HY_X_all'] = area_HY_X_all[a]
                        results['area_models'][a]['C'] = area_models[a].C_
                        results['area_models'][a]['selected_parents'] = area_best_set




            # save data 
            pickle_object(results, DI_file_path)
        pickle_object(rates, self.dataset.spiking_rates_path(self.times))



    @staticmethod
    def get_n_params(clf):
        return np.sum(np.abs(clf.coef_.squeeze()) > 0)


    @staticmethod
    def H(probs, y_true):
        return - np.sum( np.log(probs[:,1])* y_true  + np.log(probs[:,0]) * (1. - y_true))   / len(y_true)


   



def main(config, all_areas, area_filter, calculate_rates):
    
    for dataset in config.datasets:
    # for dataset, prefiltered_dataset in zip(datasets, prefiltered_datasets):
        
        ## Load the data 
        
        print('Loading dataset {}'.format(dataset))
        data = pd.read_feather(dataset.preprocessed_path)
        stimuli = config.stimuli or list(data.stimn.unique())

        # random.shuffle(stimuli)
        for stimulus_id in stimuli:
            LassoParentSelection(config, dataset, data, stimulus_id=int(stimulus_id), all_areas=all_areas, area_filter=area_filter, calculate_rates=calculate_rates)()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for parent sets')
    parser.add_argument('--config', action='store', type=str, default='../configs/default_config.yml')
    parser.add_argument('--area-filter', nargs='+', help='Restricts parent search by area', required=False, type=str, default=None)
    parser.add_argument('--all-areas', nargs='+', help='List of all areas', required=False, type=str, default=None)
    parser.add_argument('--rates', type=bool, default=False, help='Calculate rates of units instead of selecting their parent')

    args = parser.parse_args()
    
    ## Pull configuration 
    conf = Configuration(args.config)
    
    main(conf, args.all_areas, args.area_filter, args.rates)


