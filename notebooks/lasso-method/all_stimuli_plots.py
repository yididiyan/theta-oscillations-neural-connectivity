   
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

from analysis_utils import get_adjacency

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams.update({'font.size': 20})




freq = ['0.015_0.75_pinknoise.mp4',
 '0.015_1.5_pinknoise.mp4',
 '0.015_12_pinknoise.mp4',
 '0.015_3_pinknoise.mp4',
 '0.015_6_pinknoise.mp4',
 '0.03_0.75_pinknoise.mp4',
 '0.03_1.5_pinknoise.mp4',
 '0.03_12_pinknoise.mp4',
 '0.03_3_pinknoise.mp4',
 '0.03_6_pinknoise.mp4',
 '0.06_0.75_pinknoise.mp4',
 '0.06_1.5_pinknoise.mp4',
 '0.06_12_pinknoise.mp4',
 '0.06_3_pinknoise.mp4',
 '0.06_6_pinknoise.mp4',
 '0.12_0.75_pinknoise.mp4',
 '0.12_1.5_pinknoise.mp4',
 '0.12_12_pinknoise.mp4',
 '0.12_3_pinknoise.mp4',
 '0.12_6_pinknoise.mp4',
 '0.24_0.75_pinknoise.mp4',
 '0.24_1.5_pinknoise.mp4',
 '0.24_12_pinknoise.mp4',
 '0.24_3_pinknoise.mp4',
 '0.24_6_pinknoise.mp4']

spatial_freqs = [1.5, 3, 6, 12, 24]
temporal_freqs = [0.75, 1.5, 3.0, 6.0, 12.0]
groups = areas = ['AL', 'LM', 'V1']


def area_based_diff(area_1, area_2, files_pre, files_post, rates_path=None, **kwargs):
    '''
    Diff plot based on a single area, and the row values for 
    '''
    
    areas = kwargs.get('areas', ['AL', 'LM', 'V1'])
    stimuli = kwargs.get('stimuli', range(25))
    
    area_1_index = areas.index(area_1)
    area_2_index = areas.index(area_2)
    
    diff_values = [] 
    raw_values = {'pre': {}, 'post': {}}
    rates_path = rates_path or {'pre': None, 'post': None}
    
    for j in stimuli:
        
        stimulus_id = j
        Adj_median_pre, Adj_pre = get_adjacency(files_pre, stimulus_id, rates_path['pre'], **kwargs)
        Adj_median_post, Adj_post = get_adjacency(files_post, stimulus_id, rates_path['post'], **kwargs)
        
        diff = Adj_median_post - Adj_median_pre
        

        raw_values['pre'][stimulus_id] = Adj_pre[area_1][area_2]
        raw_values['post'][stimulus_id] = Adj_post[area_1][area_2]

        diff_values.append(diff[area_1_index][area_2_index])


    diff_values = np.array(diff_values)
    
    row_col_size = int(np.sqrt(len(stimuli)))
    diff_values = diff_values.reshape((row_col_size, row_col_size)).T

    return diff_values, raw_values
    

def plot_area_based_diff_plot(data, title, pvals=None, include_values=True, highlight_cells=[]):
    """
    all 25 stimuli plotted together
    :param include_values - include connectivity values in the plots as text 
    """
    fig, ax = plt.subplots( figsize=(10,10))
    vmax = np.nanmax(np.abs(data))
    vmin = -vmax
    print(f'vmax {vmax}')

#     import ipdb; ipdb.set_trace()
    im = ax.imshow(data, cmap='coolwarm', vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, 
                              ax = ax,
                              shrink=0.5 )
    # add tick labels
    ax.set_yticks(np.arange(len(temporal_freqs)), labels=temporal_freqs)
    ax.set_xticks(np.arange(len(spatial_freqs)), labels=spatial_freqs)
    ax.set_xlabel(r'SF(cpd, x$10^{-2}$ Hz)')
    ax.set_ylabel('TF(Hz)')

    # Rotate the tick labels to be more legible
    plt.setp(ax.get_xticklabels(),
             rotation = 45,
             ha = "right",
             rotation_mode = "anchor")
    ax.set_title(title, size=20)
    fig.tight_layout()
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = ''
            if include_values:
                txt = '%.2f' % data[i, j]
            if pvals is not None:
                if pvals[i, j] < 0.001:
                    txt += '\n***'
                elif pvals[i, j] < 0.01:
                    txt += '\n**'
                elif pvals[i, j] < 0.05:
                    txt += '\n*'
                
            text = ax.text(j, i, txt,
                           ha="center", va="center", color="w")
    
    ## highlight important cells 
    if highlight_cells:
        for x,y in highlight_cells:
            rect = plt.Rectangle((y-.5, x-.5), 1,1, fill=False, edgecolor='#00FF90', linewidth=4)
            ax.add_patch(rect)
    
    return fig



def read_plot_data(base_dir, files_pre, files_post, **kwargs):
    raw, data = {}, {}
    areas_ = kwargs.get('areas', areas)

    for from_ in areas_:
        data[from_], raw[from_] = {}, {}
        for to in areas_:
            data[from_][to], raw[from_][to]  = area_based_diff(from_, to, files_pre, files_post, **kwargs)
    
    return {
        'raw': raw,
        'data': data
    }






def plot_all_area_based(data, stimulus_id, output, n_runs=10**5, pvals=None, include_values=True, highlight_cells=[], area_filters=None):
    """
    :param include_values - include connectivity values in the plots as text 
    """
    pdf = matplotlib.backends.backend_pdf.PdfPages(output)


    for from_ in areas:
        for to in areas:
            pvals_ = None 
            if pvals:
                pvals_ = pvals[from_][to]
                pvals_ = np.array(pvals_).reshape((5, 5)).T
            if area_filters and (from_, to) not in area_filters:
                continue
            
            fig = plot_area_based_diff_plot(data['data'][from_][to], f'Stimulus {stimulus_id} - {from_} -> {to}', pvals_, include_values=include_values, highlight_cells=highlight_cells)
            pdf.savefig(fig)

    pdf.close()




def plot_heatmaps_with_all_stimuli(pre_files, post_files, post_title, pre_title='Pre', stimuli=None, output='default_output.pdf', rates_path=None, min_rate=0.):
    """
    Area to area under each stimuli  ---> ( area X area ) shaped plots 
    """

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




        axes[0].set_title(pre_title)
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



if __name__ == '__main__':
    import pickle 

    # directory where the output folder resides 
    # path to generated DI files -- see example below 
    #  
    base_dir = '../../../' 
    # import ipdb; ipdb.set_trace()
    data = read_plot_data(f'{base_dir}/', 
                 f'{base_dir}/output_stimuli_9/DI_values/pre_spk_times/9/500_1500/lasso_0.5/*.pkl', 
                f'{base_dir}/output_stimuli_9/DI_values/post_9_spk_times/9/500_1500/lasso_0.5/*.pkl')
    with open('./results.pkl', 'wb') as f:
        pickle.dump(data, f)

    


