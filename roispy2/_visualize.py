import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

# def _plot_rfs(df_sta, rftype, cntrtype):
    
#     # df = df_sta[['rec_id', 'roi_id', rftype, cntrtype]]
    
#     rec_ids = df_sta['rec_id']
#     roi_ids = df_sta['roi_id']
#     rfs = df_sta['{}'.format(rftype)]
#     rf_quality = df_sta['rf_quality']
#     # rf_quality_index = df_sta['rf_quality_index']
    

#     if 'upsampled' in rftype:
#         rf_cntr = df_sta['{}'.format(cntrtype[:-9] + 'real_cntr')]
#         rf_sizes = df_sta['{}_size'.format(rftype)]
#     else:
#         rf_cntr = df_sta['{}'.format(cntrtype)]
#         rf_sizes = df_sta['{}_size'.format(rftype)]
    
#     num_full_pages = len(df_sta)//32
#     remainder = len(df_sta) % 32
#     if remainder > 0:
#         num_pages = num_full_pages + 1
#     else:
#         num_pages = num_full_pages
        
#     counter = 0 
#     fig_list = []

#     for page_id in np.arange(num_pages):
        
#         fig, ax = plt.subplots(8, 4, figsize=((8.27, 11.69))) 
#         ax = ax.flatten()
#         num_subplots = len(ax)

#         for fig_idx, row in enumerate(rfs[counter*32:32+counter*32]):
            
#             idx = fig_idx + counter * 32
#             rf = row

#             rf_size = rf_sizes[idx]
#             # rf_q = rf_quality_index[idx]

#             ax[fig_idx].imshow(rf, origin='lower', cmap=plt.cm.binary)
#             ax[fig_idx].axis('off')
            
#             if not np.isnan(rf_cntr[idx]).any() and rf_size > 1:

#                 cntr_x = rf_cntr[idx][:, 0] 
#                 cntr_y = rf_cntr[idx][:, 1]
#                 ax[fig_idx].plot(cntr_y, cntr_x, color='red')
            
#             ax[fig_idx].set_title('{}({},{}): RF size {:.3f}'.format(idx, rec_ids[idx], roi_ids[idx], rf_size), fontsize=10)
            
#             if not rf_quality[idx]:   
#                 for axis in ['top','bottom','left','right']:
#                     ax[fig_idx].spines[axis].set_linewidth(3)
#                     ax[fig_idx].spines[axis].set_color('red')  
            

#         if fig_idx < 31:
#             for redundent in range(fig_idx, 32):
#                 ax[redundent].axis('off')
#         counter +=1
#         fig_list.append(fig)

#         if num_pages > 1:
#             plt.suptitle(rftype + ' ({})'.format(page_id+1))
#         else:
#             plt.suptitle(rftype)

#     return fig_list


# def _plot_rfs(df_sta):
    
#     rec_ids = df_sta['rec_id']
#     roi_ids = df_sta['roi_id']
    
#     rfs=  df_sta['sRF_asd_upsampled']
#     rf_cntrs = df_sta['sRF_asd_upsampled_cntr']
#     rf_sizes = df_sta['sRF_asd_upsampled_size']

#     highlight = np.where(~df_sta['cntr_quality'])[0]
    
#     num_full_pages = len(df_sta)//32
#     remainder = len(df_sta) % 32
#     if remainder > 0:
#         num_pages = num_full_pages + 1
#     else:
#         num_pages = num_full_pages
        
#     counter = 0 
#     fig_list = []

#     for page_id in np.arange(num_pages):
        
#         fig, ax = plt.subplots(8, 4, figsize=((8.27, 11.69))) 
#         ax = ax.flatten()
#         num_subplots = len(ax)

#         for fig_idx, row in enumerate(rfs[counter*32:32+counter*32]):
            
#             idx = fig_idx + counter * 32
#             rf = row

#             rf_size = rf_sizes[idx]
#             rf_cntr = rf_cntrs[idx]

#             # ax[fig_idx].imshow(rf, origin='lower', cmap=plt.cm.binary)

#             ax[fig_idx].imshow(rf, cmap=plt.cm.binary)
            
#             if highlight is not None and idx in highlight:
#                 for axis in ['top','bottom','left','right']:
#                     ax[fig_idx].spines[axis].set_linewidth(3)
#                     ax[fig_idx].spines[axis].set_color('red')   
#                     ax[fig_idx].axes.get_xaxis().set_visible(False)
#                     ax[fig_idx].axes.get_yaxis().set_visible(False)
#             else:
#                 ax[fig_idx].axis('off')
            
#             if len(rf_cntr)>0 and idx not in highlight:
#                 for cntr in rf_cntr:

#                     cntr_x = cntr[:, 0] 
#                     cntr_y = cntr[:, 1]
#                     ax[fig_idx].plot(cntr_y, cntr_x, color='red')
#                     ax[fig_idx].set_title('{}({},{}): RF size {:.3f}'.format(idx, rec_ids[idx], roi_ids[idx], np.sum(rf_size)), fontsize=10)

#         if fig_idx < 31:
#             for redundent in range(fig_idx, 32):
#                 ax[redundent].axis('off')
#         counter +=1
#         fig_list.append(fig)

#         if num_pages > 1:
#             plt.suptitle('upsampled ASD' + ' ({})'.format(page_id+1))
#         else:
#             plt.suptitle('upsampled ASD')

#     return fig_list   

# def _plot_rfs(df_sta):
    
#     rec_ids = df_sta['rec_id']
#     roi_ids = df_sta['roi_id']
    
#     rfs=  df_sta['sRF_asd_upsampled']
#     rf_cntrs = df_sta['sRF_asd_upsampled_cntr']
#     rf_sizes = df_sta['sRF_asd_upsampled_size']

#     highlight = np.where(~df_sta['cntr_quality'])[0]
    
#     num_full_pages = len(df_sta)//32
#     remainder = len(df_sta) % 32
#     if remainder > 0:
#         num_pages = num_full_pages + 1
#     else:
#         num_pages = num_full_pages
        
#     counter = 0 
#     fig_list = []

#     for page_id in np.arange(num_pages):
        
#         fig, ax = plt.subplots(8, 4, figsize=((8.27, 11.69))) 
#         ax = ax.flatten()
#         num_subplots = len(ax)

#         for fig_idx, row in enumerate(rfs[counter*32:32+counter*32]):
            
#             idx = fig_idx + counter * 32
#             rf = row

#             rf_size = rf_sizes[idx]
#             rf_cntr = rf_cntrs[idx]

#             # ax[fig_idx].imshow(rf, origin='lower', cmap=plt.cm.binary)

#             ax[fig_idx].imshow(rf, cmap=plt.cm.binary)
            
#             if highlight is not None and idx in highlight:
#                 for axis in ['top','bottom','left','right']:
#                     ax[fig_idx].spines[axis].set_linewidth(3)
#                     ax[fig_idx].spines[axis].set_color('red')   
#                     ax[fig_idx].axes.get_xaxis().set_visible(False)
#                     ax[fig_idx].axes.get_yaxis().set_visible(False)
#             else:
#                 ax[fig_idx].axis('off')
            
#             if len(rf_cntr)>0 and idx not in highlight:
#                 for cntr in rf_cntr:

#                     cntr_x = cntr[:, 0] 
#                     cntr_y = cntr[:, 1]
#                     ax[fig_idx].plot(cntr_y, cntr_x, color='red')
#                     ax[fig_idx].set_title('{}({},{}): RF size {:.3f}'.format(idx, rec_ids[idx], roi_ids[idx], np.sum(rf_size)), fontsize=10)

#         if fig_idx < 31:
#             for redundent in range(fig_idx, 32):
#                 ax[redundent].axis('off')
#         counter +=1
#         fig_list.append(fig)

#         if num_pages > 1:
#             plt.suptitle('upsampled ASD' + ' ({})'.format(page_id+1))
#         else:
#             plt.suptitle('upsampled ASD')

#     return fig_list   

def _plot_rfs(df_sta, df_cntr, kind='sRF_upsampled'):
    
    def rescale_data(data):
        return (data - data.min()) / (data.max() - data.min())
    
    rec_ids = df_sta['rec_id']
    roi_ids = df_sta['roi_id']
    
    rfs = df_sta[kind]
    
    num_row = 6
    num_col = 4
    num_rf_each_page = num_row * num_col
    
    num_full_pages = len(df_sta)//num_rf_each_page
    remainder = len(df_sta) % num_rf_each_page
    if remainder > 0:
        num_pages = num_full_pages + 1
    else:
        num_pages = num_full_pages
        
    counter = 0 
    fig_list = []
    
    # levels=np.linspace(0, 1, 41)[::2][10:-6]
    # levels = np.arange(55, 75, 5)/100
    levels = np.arange(60, 75, 5)/100

    quality = df_cntr['cntr_quality']

    for page_id in np.arange(num_pages):
        
        fig, ax = plt.subplots(num_row, num_col, figsize=((8.27, 11.69))) 
        ax = ax.flatten()
        num_subplots = len(ax)

        for fig_idx, row in enumerate(rfs[counter*num_rf_each_page:num_rf_each_page+counter*num_rf_each_page]):
            
            idx = fig_idx + counter * num_rf_each_page
            rf = row
            
            # mask = df_cntr[['cntr_quality_50', 
            #                   'cntr_quality_55', 
            #                   'cntr_quality_60', 
            #                   'cntr_quality_65', 
            #                   'cntr_quality_70']].loc[idx].values
        
            if quality[idx] == False: 
                
                ax[fig_idx].imshow(rf, cmap=plt.cm.binary, origin='lower')
                ax[fig_idx].set_title('{}({},{})'.format(idx, rec_ids[idx], roi_ids[idx]), fontsize=10)
                ax[fig_idx].contour(rescale_data(rf), levels=levels)
                # C = ax[fig_idx].contour(rescale_data(rf), levels=levels)
                # ax[fig_idx].clabel(C, inline=True, fontsize=10)
                ax[fig_idx].set_title('{}({},{})'.format(idx, rec_ids[idx], roi_ids[idx]), fontsize=10)
                for axis in ['top','bottom','left','right']:
                    ax[fig_idx].spines[axis].set_linewidth(3)
                    ax[fig_idx].spines[axis].set_color('red') 
                    ax[fig_idx].axes.get_xaxis().set_visible(False)
                    ax[fig_idx].axes.get_yaxis().set_visible(False)

                continue

            ax[fig_idx].imshow(rf, cmap=plt.cm.binary, origin='lower')
            C = ax[fig_idx].contour(rescale_data(rf), levels=levels)
            # ax[fig_idx].clabel(C, inline=True, fontsize=10)
            ax[fig_idx].set_title('{}({},{})'.format(idx, rec_ids[idx], roi_ids[idx]), fontsize=10)
            ax[fig_idx].axis('off')

            
        if fig_idx < num_rf_each_page-1:
            for redundent in range(fig_idx+1, num_rf_each_page):
                ax[redundent].axis('off')
        counter +=1
        fig_list.append(fig)

        if num_pages > 1:
            plt.suptitle('upsampled SPL' + ' ({})'.format(page_id+1))
        else:
            plt.suptitle('upsampled SPL')

    return fig_list   