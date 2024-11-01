# updated utilities from /notebooks/lasso/calculate_pvals.py
import numpy as np 
from multiprocessing import Pool
import sys 


from statsmodels.stats.multitest import multipletests as mt 



sys.path.append('../notebooks/lasso-method/')
sys.path.append('../code/')



from calculate_pvalues import one_run_sample
import matplotlib.pyplot as plt 



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
#     print(tmp, len(vals), true_diff)
    return sum(tmp)/len(vals)

def calculate_p_value(pre_connections, post_connections, true_diff, n_runs=10**6):
    

    pre_counts = len(pre_connections)
    post_counts = len(post_connections)

    if pre_counts + post_counts == 0:
        return np.nan 

    pre_prop = pre_counts * 1. / (pre_counts + post_counts)
    print('Proportion |Pre| / (|Pre| + |Post|)', pre_prop)

    pval = get_pval_diff_med(pre_connections, post_connections, pre_prop, true_diff, n_runs=n_runs)

    if pval < 0.05:
        print(f'Significant - pre - {pre_counts}, post - {post_counts} with significance value {pval}')


    return pval      



## plots 

def plot_per_training(data_, title, training='pre', areas=[], stimulus_id=0, max_=8.):

    group_matrix = {}
    median_matrix = np.nan * np.ones((len(areas), len(areas)))
    edge_matrix = np.nan * np.ones((len(areas), len(areas)))

    for f, from_area in enumerate(areas):
        for t, to_area in enumerate(areas):
            print(f'\nFrom {from_area} to {to_area}')
            connections = data_['raw'][from_area][to_area][training][stimulus_id]

            median_val = np.median(connections)
            edge_val = len(connections)
            

            group_matrix[(from_area, to_area)] = (f,t) 
            median_matrix[(f, t)] = median_val
            edge_matrix[(f,t)] = edge_val


   

    plt.imshow(median_matrix, interpolation='none',
               cmap='Reds',
               vmin=0, 
               vmax=max_)
    plt.title(title)
    plt.xticks(np.arange(len(areas)), labels=[ a.upper() for a in areas])
    plt.xlabel('To')
    plt.ylabel('From')
    plt.yticks(np.arange(len(areas)), labels=[ a.upper() for a in areas])
    
    
    plt.colorbar()
    
    for i  in range(median_matrix.shape[0]):
        for j in range(median_matrix.shape[1]):
            if median_matrix[i][j] > max_ / 2:
                plt.text(j, i, f'{median_matrix[i][j]:.2f}\n({int(edge_matrix[i][j])})', ha="center", va="center", color="w")
            else:
                plt.text(j, i, f'{median_matrix[i][j]:.2f}\n({int(edge_matrix[i][j])})', ha="center", va="center", color="black")
            


def plot(data_, title, corrected=False, areas=[], stimulus_id=0, max_=4., n_runs=10**6):
    

    pval_matrix = np.nan * np.ones((len(areas), len(areas)))
    group_matrix = {}
    true_diff_matrix = np.nan * np.ones((len(areas), len(areas)))

    for f, from_area in enumerate(areas):
        for t, to_area in enumerate(areas):
            print(f'\nFrom {from_area} to {to_area}')
            pre_connections = data_['raw'][from_area][to_area]['pre'][stimulus_id]
            post_connections = data_['raw'][from_area][to_area]['post'][stimulus_id]

            true_diff= data_['data'][from_area][to_area]
            print('true_diff ', true_diff)
            pval = calculate_p_value(pre_connections, post_connections, true_diff, n_runs=n_runs  )



            group_matrix[(from_area, to_area)] = (f,t) 
            true_diff_matrix[(f, t)] = true_diff
            pval_matrix[(f,t)] = pval


    plt.imshow(true_diff_matrix, interpolation='none',
               cmap=plt.cm.coolwarm,
               vmin=-max_, 
               vmax=max_)
    plt.title(title)
    plt.xticks(np.arange(len(areas)), labels=[ a.upper() for a in areas])
    plt.xlabel('To')
    plt.ylabel('From')
    plt.yticks(np.arange(len(areas)), labels=[ a.upper() for a in areas])

    pvals_flat = pval_matrix.reshape(pval_matrix.shape[0] * pval_matrix.shape[1])
    if corrected:
        pvals_flat = mt(pvals_flat, method='fdr_bh')[1]

    print(f'pvals {pvals_flat}')
    pval_matrix = pvals_flat.reshape(pval_matrix.shape[0], pval_matrix.shape[1])
    
    for i  in range(pval_matrix.shape[0]):
        for j in range(pval_matrix.shape[1]):
            if pval_matrix[(i,j)] < 0.001:
                plt.text(j, i, '***', ha="center", va="center", color="black")
            elif pval_matrix[(i,j)] < 0.01:
                plt.text(j, i, '**', ha="center", va="center", color="black")
            elif pval_matrix[(i,j)] < 0.05:
                plt.text(j, i, '*', ha="center", va="center", color="black")

    plt.colorbar()