import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

def _plot_rfs(df_sta, rftype, cntrtype):
    
    # df = df_sta[['rec_id', 'roi_id', rftype, cntrtype]]
    
    rec_ids = df_sta['rec_id']
    roi_ids = df_sta['roi_id']
    rfs = df_sta['{}'.format(rftype)]
    

    if 'upsampled' in rftype:
        rf_cntr = df_sta['{}'.format(cntrtype[:-9] + 'real_cntr')]
        rf_sizes = df_sta['{}_size'.format(rftype)]
    else:
        rf_cntr = df_sta['{}'.format(cntrtype)]
        rf_sizes = df_sta['{}_size'.format(rftype)]
    
    num_full_pages = len(df_sta)//32
    remainder = len(df_sta) % 32
    if remainder > 0:
        num_pages = num_full_pages + 1
    else:
        num_pages = num_full_pages
        
    counter = 0 
    fig_list = []

    for page_id in np.arange(num_pages):
        
        fig, ax = plt.subplots(8, 4, figsize=((8.27, 11.69))) 
        ax = ax.flatten()
        num_subplots = len(ax)

        for fig_idx, row in enumerate(rfs[counter*32:32+counter*32]):
            
            idx = fig_idx + counter * 32
            rf = row
            rf_size = rf_sizes[idx]
            
            ax[fig_idx].imshow(rf, origin='lower', cmap=plt.cm.binary)
            ax[fig_idx].axis('off')
            
            if not np.isnan(rf_cntr[idx]).any() and rf_size > 1:

                cntr_x = rf_cntr[idx][:, 0] 
                cntr_y = rf_cntr[idx][:, 1]
                ax[fig_idx].plot(cntr_y, cntr_x, color='red')
            
            ax[fig_idx].set_title('{}({},{}): RF size {:.3f}'.format(idx, rec_ids[idx], roi_ids[idx], rf_size), fontsize=10)

        
        if fig_idx < 31:
            for redundent in range(fig_idx, 32):
                ax[redundent].axis('off')
        counter +=1
        fig_list.append(fig)

        if num_pages > 1:
            plt.suptitle(rftype + ' ({})'.format(page_id+1))
        else:
            plt.suptitle(rftype)

    return fig_list