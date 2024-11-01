import pickle 
import itertools
import os
import sys 
import glob 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 
import matplotlib.backends.backend_pdf

sys.path.append('../../code/')
sys.path.append('../../analysis/')
# import utils




plt.rcParams['figure.figsize'] = (25, 25)
plt.rcParams.update({'font.size': 20})





groups = ['AL', 'LM', 'V1']




def read_pickle_file(filename):
    try: 
        with open(filename, 'rb') as f:
            return pickle.load(f)

    except EOFError as e:
        print('Corrupted pickle ', filename)
        print(e) ## dump error 

        os.remove(filename)
        print('Deleted pickle file successfully:  ', filename)
    except:
        return None

def pickle_object(object, filename):
    directory = str(Path(filename).parent)
    os.makedirs(directory, exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(object, f)


def get_adjacency(file_dir, stimulus_id, rates_path=None, min_rate=0., percentile=95, use_ate=False, **kwargs):
    '''
    
    Filter units by rates, only considers units with at least min_rate of spiking rate. 
    percentile  - firing rate cutoff 
    '''
    # load files 
    if type(file_dir) == str:
        files = glob.glob(file_dir)
    else:
        files = file_dir # list of files 
    
    issues = {
        'negative_di': 0,
        'n_areas': 0,
        'zero_contrib': 0,
        'low_firing_rates': 0
    }

    multi_area = kwargs.get('multi_area', None)
    areas = kwargs.get('areas', ['AL', 'LM', 'V1'])
    rates = read_pickle_file(rates_path)
    if rates:
        min_rate = min_rate or np.percentile(list(rates.values()), percentile)
        print(f'Min rate {min_rate}')

#     import ipdb; ipdb.set_trace()
    # prepare adjancency 
    Adj = {}
    for groupX in areas:
        Adj[groupX] = {}
        for groupY in areas:
            Adj[groupX][groupY] = []

    results = []
    print(f'Working with {len(files)} units for stimulus {stimulus_id} with min spiking rate of {min_rate}', end='\r')
    
    for f in files:
        
        ## check if the unit has enough spikes 
        Yunit_id = Path(f).stem.replace('DI_dict_', '')
        
        if rates: 
            firing_rate = rates.get(Yunit_id, None)
            if firing_rate and firing_rate < min_rate:
                issues['low_firing_rates']+=1
                continue 
                
            
        groupY = Path(f).stem.split('_')[-2]

        result = None
        data = read_pickle_file(f)
        
        if stimulus_id != 'all' and stimulus_id not in data['HY'].keys():
            continue
        
        if use_ate:
            if 'ATE' in data.keys():
                for a, ate_value in data['ATE'][stimulus_id].items():
                    Adj[a][groupY].append(ate_value)

        else: 
            di =  data['DI_Y_X_all'] if stimulus_id == 'all' else data['DI_Y_X'][stimulus_id]
            HY = data['HY_all'] if stimulus_id == 'all' else data['HY'][stimulus_id]
            HY_X = data['HY_X_all'] if stimulus_id == 'all' else data['HY_X'][stimulus_id]
            
            
            if 'area_models' not in data.keys():
                issues['negative_di'] += 1
                continue
                
            if di > 0: 
                result = {}
                areas_ = sorted(list(set([x.split('_')[-2] for x in data['selected_parents']])))

                normalized_value = (di / HY) * 100. ## percentage entropy reduction 

                # shapely like marginal gain calculation marginal gain -- averaged
                
                contributions = {a: None for a in areas_}

                if multi_area:
                    # A multi-area version of Shapley value calculation 
                    
                    
                    area_combos = [c for i in range(1, len(areas_)) for c in list(itertools.combinations(areas_, i)) ]
                   
                    for a in areas_:
                        # get all combos containing a 
                        # TODO: THINK ABOUT WHEN THERE ARE CASES where areas_ != all_areas ==> not the complete tree of combinations
                        contributions_a = []
                        weights_a = []

                        for i in range(0, len(areas_)):
                            if i == 0:
                                # compare it to the plain model 
                                contributions_a.append(max(HY - 
                                            (data['area_models'][(a,)]['HY_X_all']  if stimulus_id == 'all' else data['area_models'][(a,)]['HY_X'][stimulus_id])
                                        , 0))
                                weights_a.append(1.)

                            else:
                                
                                # get all the combinations in the current layer that don't contain the area "a"
                                combos_filtered = [c for c in area_combos if len(c) == i and (a not in c) ]
                                layer_weight = 1. / len(combos_filtered)  # the weight assigned to a node in the layer 

                                for c in combos_filtered:
                                    # grab the combinations 
                                    layer_combo = c
                                    # add the newly added area "a"
                                    next_layer_combo = list(layer_combo) + [a]
                                    next_layer_combo = tuple(sorted(next_layer_combo)) # in the order the areas are defined 
                                    if stimulus_id == 'all':
                                        next_layer_entropy = data['area_models'][next_layer_combo]['HY_X_all'] if len(next_layer_combo) < len(areas_) else HY_X
                                        contributions_a.append(
                                            max(data['area_models'][layer_combo]['HY_X_all'] - next_layer_entropy, 0)
                                        )
                                        weights_a.append(layer_weight)
                                    else:
                                        next_layer_entropy = data['area_models'][next_layer_combo]['HY_X'][stimulus_id] if len(next_layer_combo) < len(areas_) else HY_X
                                        contributions_a.append(
                                            max(data['area_models'][layer_combo]['HY_X'][stimulus_id] - next_layer_entropy, 0)
                                        )
                                        weights_a.append(layer_weight)
                        
                        # contributions_a = contributions_a * weights 
                        # import ipdb; ipdb.set_trace()
                        contributions_a = np.array(contributions_a) / np.array(weights_a)
                        contributions[a] = contributions_a.sum() / len(areas_)
                        

                    
                    
                    
                else:
                    # # TODO: addressing cases where only one area is involved 
                    # if len(areas_) == 1:
                    #     contributions[areas_[0]]  = max(HY - data['area_models'][areas_[0]]['HY_X'][stimulus_id], 0)
                    # el
                    if len(areas_) == 2:
                        
                        if stimulus_id == 'all':
                            contributions[areas_[0]] = sum([
                                    max(HY - data['area_models'][areas_[0]]['HY_X_all'], 0),
                                    max(data['area_models'][areas_[1]]['HY_X_all'] - HY_X, 0)
                                        ]) / 2.

                            contributions[areas_[1]] = sum([
                                    max(HY - data['area_models'][areas_[1]]['HY_X_all'], 0),
                                    max(data['area_models'][areas_[0]]['HY_X_all'] - HY_X, 0)
                                        ]) / 2.
                        else:
                            contributions[areas_[0]] = sum([
                                    max(HY - data['area_models'][areas_[0]]['HY_X'][stimulus_id], 0),
                                    max(data['area_models'][areas_[1]]['HY_X'][stimulus_id] - HY_X, 0)
                                        ]) / 2.

                            contributions[areas_[1]] = sum([
                                    max(HY - data['area_models'][areas_[1]]['HY_X'][stimulus_id], 0),
                                    max(data['area_models'][areas_[0]]['HY_X'][stimulus_id] - HY_X, 0)
                                        ]) / 2.
                    else:
                        continue    
                
                total_contribution = sum([c for c in contributions.values()])

                if total_contribution <= 0.:
                    print('zero total contribution')
                    issues['zero_contrib'] += 1
                    continue
                
                ## normalize our contributions 
                contributions = {a: normalized_value * contributions[a]/total_contribution for a in  areas_ }
                
                result['contributions'] = contributions
                result['normalized_di'] = normalized_value
                for c, v in contributions.items():
                    if v > 0.:  Adj[c][groupY].append(v) ## only account for positive contributions 
                    # Adj[c][groupY].append(v)        
                
    
    
    
    

    Adj_median = np.zeros((len(areas), len(areas)))
    for i, groupX in enumerate(areas):
        for j, groupY in enumerate(areas):
            Adj_median[i,j] = np.median(Adj[groupX][groupY])
            
            
    print(f'Issues summary {issues}')
    return Adj_median, Adj 




def triangular_adjacency(Adj):
    '''
    Deprecated: to be removed 
    Returns the triangular version of the adjacency 
    '''
    areas = ['AL', 'LM', 'V1']
    
    for groupX in areas:
        for groupY in areas:
            if groupX != groupY:
                Adj[groupX][groupY] += Adj[groupY][groupX]
                Adj[groupX][groupY] = sorted(Adj[groupX][groupY])
                Adj[groupY][groupX] = []
    Adj_median = np.zeros((len(areas), len(areas)))
    for i, groupX in enumerate(areas):
        for j, groupY in enumerate(areas):
            Adj_median[i,j] = np.median(Adj[groupX][groupY])
            
    return Adj_median, Adj


    




def plot_heatmaps_with_all_stimuli(pre_files, post_files, post_title, pre_title='Pre', stimuli=None, output='default_output.pdf', rates_path=None, min_rate=0.):

    pdf = matplotlib.backends.backend_pdf.PdfPages(output)

    
    stimuli = stimuli or list(range(25))

    rates_path = rates_path or {'pre': None, 'post': None} # default rates 

    for i, j in enumerate(stimuli):
        fig, axes = plt.subplots(1, 3)
        
        ## Stimulus 9 
        stimulus_id = j

        Adj_median_pre, Adj_pre = get_adjacency(pre_files, stimulus_id, min_rate=min_rate, rates_path=rates_path['pre'])
        Adj_median_post, Adj_post_9 = get_adjacency(post_files, stimulus_id, min_rate=min_rate, rates_path=rates_path['post'])

        diff = Adj_median_post - Adj_median_pre





#         print(f' Pre {Adj_median_pre}')
#         print(f' Post 9 {Adj_median_post}')
#         print(f' Diff {diff}')
        axes[0].set_title('Pre')
        axes[0].imshow(Adj_median_pre, cmap='Reds',  vmin=0, vmax=3)
        axes[0].set_yticks(np.arange(len(groups)), labels=groups)
        axes[0].set_xticks(np.arange(len(groups)), labels=groups)
        axes[0].set_xlabel('to')
        axes[0].set_ylabel(f'Stimulus {stimulus_id} \nfrom')
    #     plt.show()



        axes[1].set_title(post_title)
        axes[1].imshow(Adj_median_post, cmap='Reds',  vmin=0, vmax=3)
        axes[1].set_yticks(np.arange(len(groups)), labels=groups)
        axes[1].set_xticks(np.arange(len(groups)), labels=groups)
        axes[1].set_xlabel('to')
        axes[1].set_ylabel(f'Stimulus {stimulus_id} \nfrom')
    #     plt.show()


        ## diff plot - light ones - negative, dark ones - slight rise 

        axes[2].set_title('Diff')
        axes[2].imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2].set_yticks(np.arange(len(groups)), labels=groups)
        axes[2].set_xticks(np.arange(len(groups)), labels=groups)
        axes[2].set_xlabel('to')
        axes[2].set_ylabel(f'Stimulus {stimulus_id} \nfrom')
    #     plt.show()

        plt.show()
        pdf.savefig( fig )
        
        
    pdf.close()
    



def area_plot(Adj_pre, Adj_post, area_1, area_2):
    ''''
    Histogram plots to look at distribution of area to area DI values 
    
    '''

    plt.hist(Adj_pre[area_1][area_2], label='pre', alpha=0.5)
    plt.hist(Adj_post[area_1][area_2], label='post', alpha=0.5)

    plt.axvline(np.median(Adj_pre[area_1][area_2]), color ='orange', label='pre_median')
    plt.axvline(np.median(Adj_post[area_1][area_2]), color='blue', label='post_median' )

    plt.legend()
    plt.title('{} -> {}'.format(area_1, area_2))

    plt.figure()

    plt.hist(Adj_pre[area_2][area_1], label='pre', alpha=0.5)
    plt.hist(Adj_post[area_2][area_1], label='post', alpha=0.5)

    plt.axvline(np.median(Adj_pre[area_2][area_1]), color ='orange', label='pre_median')
    plt.axvline(np.median(Adj_post[area_2][area_1]), color='blue', label='post_median' )

    plt.legend()
    plt.title('{} -> {}'.format(area_2, area_1))
    
    plt.figure()

    print('{} -> {}'.format(area_1, area_2), ' pre ', np.median(Adj_pre[area_1][area_2]), ' post ', np.median(Adj_post[area_1][area_2]))
    print('{} -> {}'.format(area_2, area_1), ' pre ', np.median(Adj_pre[area_2][area_1]), ' post ', np.median(Adj_post[area_2][area_1]))

    print('{} -> {}:  (pre: post) {}: {}'.format(area_1, area_2, len(Adj_pre[area_1][area_2]), len(Adj_post[area_1][area_2]) ))
    print('{} -> {}:  (pre: post) {}: {}'.format(area_2, area_1, len(Adj_pre[area_2][area_1]), len(Adj_post[area_2][area_1]) ))
    
    plt.show()
    


