# %%
import sys 

import os 
import logging
import itertools

import numpy as np
import pandas as pd 
import pickle 


from multiprocessing import Pool

from statsmodels.stats.multitest import multipletests as mt 


import os 
import sys 
sys.path.insert(0, os.path.abspath('../../code'))
import utils 

# %%

spatial_freqs = [1.5, 3, 6, 12, 24]
temporal_freqs = [0.75, 1.5, 3.0, 6.0, 12.0]
areas = ['AL', 'LM', 'V1']

freq_combinations = list(itertools.product(spatial_freqs, temporal_freqs))

def one_run_sample(input_dic):
    #a single run 
    prop1 = input_dic['prop']
    edges = input_dic['edges']
    vals_1 = []
    vals_2 = []
    
    
    for edge in edges:
        if np.random.binomial(1,prop1):
            vals_1.append(edge)
        else:
            vals_2.append(edge)
    
    if len(vals_1)==0:
        vals_1 = [0]
    elif len(vals_2) == 0:
        vals_2 = [0]
    
    diff = np.median(vals_1) - np.median(vals_2)

    return diff
    
    
def get_pval_diff_med(edges1,edges2,prop_1, true_diff, n_runs=10**6):
    """
    This uses Monte Carlo sampling to approximate permutation test on all possible edges (not grouping by recording)    
    """  
    if len(edges2)+ len(edges1) == 0:
        return 1.0
    
    input_dic = {'edges':list(edges1)+list(edges2),'prop':prop_1}
    print('Using %i runs'%(n_runs))
    
    vals = []
    
    pool = Pool(processes=11)
    vals = pool.map(one_run_sample, [input_dic]*n_runs)
    
    pool.close()
    pool.join()
    
    
    #Now look at one/two sided
#    print('Two sided')
    vals = np.abs(vals)
    true_diff = np.abs(true_diff)

    tmp = [1. for val in vals if val >= true_diff]
    print(tmp, len(vals), true_diff)
    return sum(tmp)/len(vals)
    

    
def calculate_p_value(pre_connections, post_connections, true_diff, n_runs=5*10**5):
    
    
    pvals = []
    
    for i, j in enumerate(freq_combinations):
        
        
        pre_counts = len(pre_connections[i])
        post_counts = len(post_connections[i])

        if pre_counts + post_counts == 0:
            pvals.append(np.nan)
            continue
        
        pre_prop = pre_counts * 1. / (pre_counts + post_counts)
        
        pval = get_pval_diff_med(pre_connections[i], post_connections[i], pre_prop, true_diff[i], n_runs=n_runs)
        print(pval)
        pvals.append(pval)
        
        if pval < 0.05:
            print(f'Significant {j}, pre - {pre_counts}, post - {post_counts} with significance value {pval}')
#         break

    return pvals      





def correct_pvalues(p_values, groups=['AL', 'LM', 'V1'], area_filter=[]):
    """
    :param p_values dict of dicts with [groupX][groupY] keys 
    
    """
    groups = ['AL', 'LM', 'V1']
    
    pvals_expanded = [ ]
    
    
    for groupX in groups:
        for groupY in groups:
            if (groupX, groupY) in area_filter:
                pvals_expanded = pvals_expanded + p_values[groupX][groupY]
            
                
                
    print(len(pvals_expanded))
    test_results = mt(pvals_expanded, method='fdr_bh')[1]
    
    corrected_pvals = {g:{ g: None for g in groups} for g in groups}
    
    index = 0
    for groupX in groups:
        for groupY in groups:
            if (groupX, groupY) in area_filter:
                corrected_pvals[groupX][groupY] = test_results[index: index + 25]
                index += 25 
            else:
                corrected_pvals[groupX][groupY] = [np.nan] * 25
    
    
    return corrected_pvals

    
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', action='store', type=str)
    parser.add_argument('--output', action='store', type=str)
    parser.add_argument('--n-runs', action='store', type=int, default=5*10**4)

    args = parser.parse_args()
    print(vars(args), args.data)
    arg_vals = vars(args)

    areas = ['AL', 'LM', 'V1']

    pvals = {}
    
    ## save plot data 
    data = utils.read_pickle_file(arg_vals['data'])

    for a1 in areas:
        pvals[a1] = {}
        for a2 in areas:
            if a1 == a2 == 'V1':
                pvals[a1][a2] =  calculate_p_value(data['raw'][a1][a2]['pre'], 
                            data['raw'][a1][a2]['post'], 
                            true_diff=np.reshape(data['data'][a1][a2].T, (5 * 5)), n_runs=args.n_runs)
    
    ## raw, uncorrected pvals 
    utils.pickle_object(pvals,args.output)


