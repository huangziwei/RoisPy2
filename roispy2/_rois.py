import os
import logging

import numpy as np
import scipy as sp

import morphopy as mp

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from ._utils import *
from ._sta import *
from ._visualize import _plot_rfs



__all__ = ['ROIs']

class ROIs:

    def __init__(self, expmeta, loglevel='info'):

        self._logger = get_logger(loglevel)

        experimenter = expmeta['experimenter']
        expdate = expmeta['expdate']
        expnum = expmeta['expnum']
        rootdir = expmeta['rootdir']
        # morphdir = expmeta['morphdir']

        self.experimenter = experimenter
        self.expdate = expdate
        self.expnum = expnum
        self.unit = expmeta['unit']
        self.data_paths = get_data_paths(rootdir, experimenter, expdate, expnum)
        

        stack_h5_path = self.data_paths['stack_h5_path']
        soma_noise_h5_path = self.data_paths['soma_noise_h5_path']  
        soma_chirp_h5_path = self.data_paths['soma_chirp_h5_path'] 
        soma_lchirp_h5_path = self.data_paths['soma_lchirp_h5_path']

        headerfile = self.data_paths['headerfile']

        with open(self.data_paths['headerfile']) as f:
            setup_id = int([line for line in f.readlines() if 'string_setupID' in line][0].strip('\n').split('=')[-1])

        if setup_id == 2:
            self.flip_rf = True
        else:
            self.flip_rf = False

        logging.info('  loading stack.h5\n')
        self.data_stack = load_h5_data(stack_h5_path)
        self.pixel_sizes_stack = get_pixel_size_stack(self.data_stack)

        # load stimulus
        logging.info('  loading noise stimulus\n')
        self.stimulus_noise = load_h5_data(expmeta['stimulus_path'] + 'noise.h5')['k'].reshape(15*20, -1)
        logging.info('  loading chirp stimulus\n')
        self.stimulus_chirp = load_h5_data(expmeta['stimulus_path'] + 'chirp.h5')['chirp']

        # load soma data
        ## the most important one is soma_noise_h5, if absent, use others instead. 
        logging.info('  loading data soma_noise.h5\n')
        if soma_noise_h5_path != None:
            self.data_soma_noise = load_h5_data(soma_noise_h5_path)
            self.data_soma_noise['type'] = 'original'
        elif soma_chirp_h5_path != None:
            logging.info('  * No soma_noise.h5, use soma_chirp.h5 instead.\n')
            self.data_soma_noise = load_h5_data(soma_chirp_h5_path)
            self.data_soma_noise['type'] = 'substitute'
        elif soma_lchirp_h5_path != None:
            logging.info('  * No soma_noise/chirp.h5, use soma_lchirp.h5 instead.\n')
            self.data_soma_noise = load_h5_data(soma_lchirp_h5_path)
            self.data_soma_noise['type'] = 'substitute'

        if soma_chirp_h5_path != None:
            logging.info('  loading data soma_chirp.h5\n')
            self.data_soma_chirp = load_h5_data(soma_chirp_h5_path)
            self.data_soma_chirp['type'] = 'original'
        if soma_lchirp_h5_path != None:
            logging.info('  loading data soma_lchirp.h5\n')
            self.data_soma_lchirp = load_h5_data(soma_lchirp_h5_path)
            self.data_soma_lchirp['type'] = 'original'

        ## Get Morph
        swc_path = self.data_paths['swc_path']
        m = mp.Morph(filepath=swc_path)
        m.summarize()

        density_maps = m.density_maps[1]
        


        self.soma = m.df_paths[m.df_paths.type == 1].path[0].flatten()
        
        self.df_paths = m.df_paths.iloc[1:]

        self.density_center = get_density_center(self.df_paths, self.soma, density_maps)
        
        # turning path from real coordinates to pixel coordinates in stack
        path_stack = self.df_paths.path.apply(lambda x: (x / self.pixel_sizes_stack).round().astype(int))
        self.df_paths = self.df_paths.assign(path_stack=pd.Series(path_stack))
        
        if 'Line_Stack_warped' in self.data_stack.keys():
            self.stack_shape = self.data_stack['Line_Stack_warped'].shape
        else:
            self.stack_shape = self.data_stack['Line_stack_warped'].shape
        self.linestack = get_linestack(self.df_paths, self.stack_shape)

    def _get_df_data(self):

        logging.info('  loading all recording data into `df_data`.\n')
        
        recording_ids = np.unique(self.df_rois['recording_id'])
        
        rec_id_dict = {}
        roi_id_dict = {}

        # NoiseArray3D_dict = {}
        
        Triggervalues_noise_dict = {}
        Triggertimes_noise_dict = {}
        Tracetimes0_noise_dict = {}
        Traces0_raw_noise_dict = {}
        Traces0_znorm_noise_dict = {}

        Averages0_chirp_dict = {}
        Triggervalues_chirp_dict = {}
        Triggertimes_chirp_dict = {}
        Tracetimes0_chirp_dict = {}
        Traces0_raw_chirp_dict = {} 
        Traces0_znorm_chirp_dict = {}
        Snippets0_chirp_dict = {}
        SnippetsTimes0_chirp_dict = {}

        Averages0_lchirp_dict = {}
        Triggervalues_lchirp_dict = {}
        Triggertimes_lchirp_dict = {}
        Tracetimes0_lchirp_dict = {}
        Traces0_raw_lchirp_dict = {} 
        Traces0_znorm_lchirp_dict = {}
        Snippets0_lchirp_dict = {}
        SnippetsTimes0_lchirp_dict = {}

        rf_s_dict = {}
        rf_t_dict = {}

        for rec_id in recording_ids:

            df_sub = self.df_rois[self.df_rois['recording_id'] == rec_id]

            filename = np.unique(df_sub['filename'])[0]
            d = load_h5_data(self.data_paths['imaging_data_dir'] + filename)

            logging.debug("  {}".format(filename))
            chirp_filename = "_".join(filename.split('_')[:3]) + '_Chirp.h5'
            
            if chirp_filename.lower() in [chirpfile.lower().split('/')[-1]
                for chirpfile in self.data_paths['chirp_h5_paths']]:
                logging.debug(" {}".format(chirp_filename))
                c = load_h5_data(self.data_paths['imaging_data_dir'] + chirp_filename)
            else:
                c = None
            
            lchirp_filename = "_".join(filename.split('_')[:3]) + '_lChirp.h5'
            
            if lchirp_filename.lower() in [lchirpfile.lower().split('/')[-1] for 
                    lchirpfile in self.data_paths['lchirp_h5_paths']]:
                logging.debug("  {}".format(lchirp_filename))
                lc = load_h5_data(self.data_paths['imaging_data_dir'] + lchirp_filename)
            else:
                lc = None        
            
            for row in df_sub.iterrows():
                
                idx = row[0]
                roi_id = int(row[1]['roi_id'])
                
                rec_id_dict[idx] = rec_id
                roi_id_dict[idx] = roi_id
                
                rf_s_dict[idx] = d['STRF_SVD_Space0'][:, :, roi_id-1]
                rf_t_dict[idx] = d['STRF_SVD_Time0'][:, roi_id-1]                  

                Triggervalues_noise_dict[idx] = d['Triggervalues']
                Triggertimes_noise_dict[idx] = d['Triggertimes']
                Tracetimes0_noise_dict[idx] = d['Tracetimes0'][:, roi_id-1]
                Traces0_raw_noise_dict[idx] = d['Traces0_raw'][:, roi_id-1] 
                Traces0_znorm_noise_dict[idx] = d['Traces0_znorm'][:, roi_id-1]

                if c:
                    Averages0_chirp_dict[idx] = c['Averages0']
                    Triggervalues_chirp_dict[idx] = c['Triggervalues']
                    Triggertimes_chirp_dict[idx] = c['Triggertimes']
                    Tracetimes0_chirp_dict[idx] = c['Tracetimes0'][:, roi_id-1]
                    Traces0_raw_chirp_dict[idx] = c['Traces0_raw'][:, roi_id-1]
                    Traces0_znorm_chirp_dict[idx] = c['Traces0_znorm'][:, roi_id-1]
                    Snippets0_chirp_dict[idx] = c['Snippets0'][:, :, roi_id-1]
                    SnippetsTimes0_chirp_dict[idx] = c['SnippetsTimes0'][:, :, roi_id-1]
                if lc:
                    Averages0_lchirp_dict[idx] = lc['Averages0']
                    Triggervalues_lchirp_dict[idx] =lc['Triggervalues']
                    Triggertimes_lchirp_dict[idx] = lc['Triggertimes']
                    Tracetimes0_lchirp_dict[idx] = lc['Tracetimes0'][:, roi_id-1]
                    Traces0_raw_lchirp_dict[idx] = lc['Traces0_raw'][:, roi_id-1]
                    Traces0_znorm_lchirp_dict[idx] = lc['Traces0_znorm'][:, roi_id-1]
                    Snippets0_lchirp_dict[idx] = lc['Snippets0'][:, :, roi_id-1]
                    SnippetsTimes0_chirp_dict[idx] = lc['SnippetsTimes0'][:, :, roi_id-1]
        
        if len([i for i in self.data_paths if 'soma_noise' in i]) != 0:
            logging.debug('  soma_noise_h5_path')
            s_n = load_h5_data(self.data_paths['soma_noise_h5_path'])
            rec_id_dict[0] = 0
            roi_id_dict[0] = 0

            rf_s_dict[0] = s_n['STRF_SVD_Space0'][:, :, 0]
            rf_t_dict[0] = s_n['STRF_SVD_Time0'][:, 0]      
            # if 'NoiseArray3D' in d.keys():
                # NoiseArray3D_dict[0] = d['NoiseArray3D']

            Triggervalues_noise_dict[0] = s_n['Triggervalues']
            Triggertimes_noise_dict[0] = s_n['Triggertimes']
            Tracetimes0_noise_dict[0] = s_n['Tracetimes0'][:, 0]
            Traces0_raw_noise_dict[0] = s_n['Traces0_raw'][:, 0]
            Traces0_znorm_noise_dict[0] = s_n['Traces0_znorm'][:, 0]
        
        soma_chirp_h5_path = [self.data_paths[i] for i in self.data_paths if 'soma_chirp' in i][0]
        if soma_chirp_h5_path is not None:
            logging.debug('soma_chirp_h5_path: {}'.format(soma_chirp_h5_path))
            s_c = load_h5_data(self.data_paths['soma_chirp_h5_path'])
            Averages0_chirp_dict[0] = s_c['Averages0']
            Triggervalues_chirp_dict[0] = s_c['Triggervalues']
            Triggertimes_chirp_dict[0] = s_c['Triggertimes']
            Tracetimes0_chirp_dict[0] = s_c['Tracetimes0'][:, 0]
            Traces0_raw_chirp_dict[0] = s_c['Traces0_raw'][:, 0]
            Traces0_znorm_chirp_dict[0] = s_c['Traces0_znorm'][:, 0]
            Snippets0_chirp_dict[0] = s_c['Snippets0'][:, :, 0]
            SnippetsTimes0_chirp_dict[0] = s_c['SnippetsTimes0'][:, :, 0]

        soma_lchirp_h5_path = [self.data_paths[i] for i in self.data_paths if 'soma_lchirp' in i][0]
        if soma_lchirp_h5_path is not None:
            logging.debug('soma_lchirp_h5_path: {}'.format(soma_lchirp_h5_path))
            s_lc = load_h5_data(self.data_paths['soma_lchirp_h5_path'])
            Averages0_lchirp_dict[0] = s_lc['Averages0']
            Triggervalues_lchirp_dict[0] =s_lc['Triggervalues']
            Triggertimes_lchirp_dict[0] = s_lc['Triggertimes']
            Tracetimes0_lchirp_dict[0] = s_lc['Tracetimes0'][:, 0]
            Traces0_raw_lchirp_dict[0] = s_lc['Traces0_raw'][:, 0]
            Traces0_znorm_lchirp_dict[0] = s_lc['Traces0_znorm'][:, 0]
            Snippets0_lchirp_dict[0] = s_lc['Snippets0'][:, :, 0]
            SnippetsTimes0_chirp_dict[0] = s_lc['SnippetsTimes0'][:, :, 0]
 
        self.df_data = pd.DataFrame()   

        self.df_data['rec_id'] = pd.Series(rec_id_dict)
        self.df_data['roi_id'] = pd.Series(roi_id_dict)
        self.df_data['rf_s'] = pd.Series(rf_s_dict)
        self.df_data['rf_t'] = pd.Series(rf_t_dict)

        # self.df_data['NoiseArray3D'] = pd.Series(NoiseArray3D_dict)

        self.df_data['Triggervalues_noise'] = pd.Series(Triggervalues_noise_dict)
        self.df_data['Triggertimes_noise'] = pd.Series(Triggertimes_noise_dict)
        self.df_data['Tracetimes0_noise'] = pd.Series(Tracetimes0_noise_dict)
        self.df_data['Traces0_raw_noise'] = pd.Series(Traces0_raw_noise_dict)
        self.df_data['Traces0_znorm_noise'] = pd.Series(Traces0_znorm_noise_dict)

        self.df_data['Averages0_chirp'] = pd.Series(Averages0_chirp_dict)
        self.df_data['Triggervalues_chirp'] = pd.Series(Triggervalues_chirp_dict)
        self.df_data['Triggertimes_chirp'] = pd.Series(Triggertimes_chirp_dict)
        self.df_data['Tracetimes0_chirp'] = pd.Series(Tracetimes0_chirp_dict)
        self.df_data['Traces0_raw_chirp'] = pd.Series(Traces0_raw_chirp_dict)
        self.df_data['Snippets0_chirp'] = pd.Series(Snippets0_chirp_dict)
        self.df_data['SnippetsTimes0_chirp'] = pd.Series(SnippetsTimes0_chirp_dict)
        
        self.df_data['Averages0_lchirp'] = pd.Series(Averages0_lchirp_dict)
        self.df_data['Triggervalues_lchirp'] = pd.Series(Triggervalues_lchirp_dict)
        self.df_data['Triggertimes_lchirp'] = pd.Series(Triggertimes_lchirp_dict)
        self.df_data['Tracetimes0_lchirp'] = pd.Series(Tracetimes0_lchirp_dict)
        self.df_data['Traces0_raw_lchirp'] = pd.Series(Traces0_raw_lchirp_dict)
        self.df_data['Snippets0_lchirp'] = pd.Series(Snippets0_lchirp_dict)
        self.df_data['SnippetsTimes0_lchirp'] = pd.Series(SnippetsTimes0_lchirp_dict)   
        
        logging.debug('\n')

    def check(self, savefig=False, save_to='./output/rois_on_trace/'):
        
        import matplotlib as mpl
        
        save_to = save_to + '{}-{}/'.format(self.expdate, self.expnum)
        noise_h5_paths = self.data_paths['noise_h5_paths']
        (stack_soma_cx, stack_soma_cy, stack_soma_cz) = self.soma / self.pixel_sizes_stack
        linestack = self.linestack
        linestack_xy = linestack.mean(2)
        
        df_rois = pd.DataFrame(columns=('recording_id', 
                                        'roi_id', 
                                        'recording_center', 
                                        'roi_pos_stack_xy',
                                        'filename'
                                        ))
        
        idx = 1
        rec_id = 1
        roi_id = 1    

        for noise_h5_path in noise_h5_paths:
            
            dname = noise_h5_path.split('/')[-1].split('_')[2]
            
            logging.info('  Reading {}'.format(dname))
            
            d = load_h5_data(noise_h5_path)
            d_rec = resize_rec(d, self.data_stack)
            d_rec_rot, (origin_shift_x, origin_shift_y) = rotate_rec(d, self.data_stack)
            d_rois_rot, roi_coords_rot = rotate_roi(d, self.data_stack)
            
            d_rel_cy, d_rel_cx, _ = rel_position_um(self.data_soma_noise, d) / self.pixel_sizes_stack[0]
            d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 
            padding = int(max(d_rec_rot.shape)) 

            crop = linestack_xy[np.maximum(0, d_stack_cx-padding):np.minimum(d_stack_cx+padding, linestack_xy.shape[0]-1), 
                                np.maximum(0, d_stack_cy-padding):np.minimum(d_stack_cy+padding, linestack_xy.shape[0]-1)]


            d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot)
            roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
            d_rois_rot_crop = np.pad(d_rois_rot, pad_width=((d_rec_rot_x0, 0), (d_rec_rot_y0, 0)), mode='constant', constant_values=255)
            d_rois_rot_crop = np.ma.masked_where(d_rois_rot_crop == 255, d_rois_rot_crop)

            rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

            roi_coords_stack_xy = roi_coords_crop + np.array([np.maximum(0, d_stack_cx-padding), 
                                                              np.maximum(0, d_stack_cy-padding)])
            d_rois_rot_stack_xy = np.pad(d_rois_rot_crop, pad_width=((np.maximum(0, d_stack_cx-padding), 0), 
                                                                     (np.maximum(0, d_stack_cy-padding), 0)), 
                                                          mode='constant', constant_values=255)
            d_rois_rot_stack_xy = np.ma.masked_where(d_rois_rot_stack_xy == 255, d_rois_rot_stack_xy)

            rec_center_stack_xy = rec_center_crop + np.array([np.maximum(0,d_stack_cx-padding), 
                                                              np.maximum(0, d_stack_cy-padding)])
            
            d_coords_xy = np.round(roi_coords_stack_xy).astype(int)


            for i, roi_xy in enumerate(d_coords_xy):

                df_rois.loc[int(idx)] = [int(rec_id), int(roi_id), rec_center_stack_xy, roi_xy, noise_h5_path.split('/')[-1]]

                idx += 1
                roi_id += 1

            roi_id = 1
            rec_id += 1
            
            ###################################
            ## Plot and check ROIs on traces ##
            ################################### 
            if savefig:

                plt.figure(figsize=(32*3/5,32))

                ax1 = plt.subplot2grid((5,3), (0,0), rowspan=2, colspan=1)
                ax2 = plt.subplot2grid((5,3), (0,1), rowspan=2, colspan=2)
                ax3 = plt.subplot2grid((5,3), (2,0), rowspan=3, colspan=3)

                # recording region
                ax1.imshow(d_rec_rot, origin='lower', cmap=plt.cm.binary)
                ax1.imshow(d_rois_rot, origin='lower', cmap=plt.cm.viridis)
                ax1.grid('off')
                ax1.scatter(roi_coords_rot[:, 1], roi_coords_rot[:, 0], color='orange', s=80)
                for point_id, point in enumerate(roi_coords_rot):
                    ax1.annotate(point_id+1, xy=point[::-1], xytext=point[::-1]-np.array([0, 2]), color='red', size=20)

                ax1.set_title('Recording Region', fontsize=24)

                # crop region
                ax2.imshow(crop, origin='lower', cmap=plt.cm.binary)
                h_d_rec_rot, w_d_rec_rot = d_rec.shape
                rect_d_rec_rot = mpl.patches.Rectangle((d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x), w_d_rec_rot, h_d_rec_rot , edgecolor='r', facecolor='none', linewidth=2)
                tmp2 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x, -d['wParamsNum'][31]) + ax2.transData
                rect_d_rec_rot.set_transform(tmp2)
                ax2.add_patch(rect_d_rec_rot)
                ax2.imshow(d_rois_rot_crop, origin='lower', cmap=plt.cm.viridis)
                ax2.scatter(roi_coords_crop[:, 1], roi_coords_crop[:, 0], s=80, color='orange')
                ax2.set_title('Cropped Region', fontsize=24)
                ax2.grid('off')

                # whole region
                ax3.imshow(linestack.mean(2), origin='lower', cmap=plt.cm.binary)
                ax3.scatter(self.soma[1]/self.pixel_sizes_stack[1], self.soma[0]/self.pixel_sizes_stack[0], s=120, marker='x')
                hd, wd = crop.shape
                rect_crop = mpl.patches.Rectangle((d_stack_cy-padding, d_stack_cx-padding), wd, hd, edgecolor='r', facecolor='none', linewidth=2)

                h_d_rec_rot, w_d_rec_rot = d_rec.shape
                rect_crop_d_rec = mpl.patches.Rectangle((d_rec_rot_y0 + d_stack_cy-padding+origin_shift_y, d_rec_rot_x0 + d_stack_cx-padding+origin_shift_x), w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
                tmp3 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+ d_stack_cy-padding+origin_shift_y, d_rec_rot_x0+d_stack_cx-padding+origin_shift_x, -d['wParamsNum'][31]) + ax3.transData
                rect_crop_d_rec.set_transform(tmp3)

                ax3.add_patch(rect_crop_d_rec)
                ax3.add_patch(rect_crop)
                ax3.imshow(d_rois_rot_stack_xy, origin='lower', cmap=plt.cm.viridis)
                ax3.scatter(roi_coords_crop[:, 1]+d_stack_cy-padding, roi_coords_crop[:, 0]+d_stack_cx-padding, s=40, color='orange')
                ax3.annotate(dname, xy=(d_rec_rot_y0 + d_stack_cy-padding-10, d_rec_rot_x0 + d_stack_cx-padding-10), color='white')
                ax3.set_title('ROIs on Cell Morpholoy', fontsize=24)
                ax3.grid('off')
                ax3.set_xlim(0,512)
                ax3.set_ylim(0,512)

                scalebar = ScaleBar(self.pixel_sizes_stack[0], units='um', location='lower left', box_alpha=0, pad=4)
                ax3.add_artist(scalebar)

                plt.suptitle('{}-{}: {}'.format(self.expdate, self.expnum, dname), fontsize=28)
                
                if not os.path.exists(save_to):
                    os.makedirs(save_to)

                plt.savefig(save_to + '{}-{}-{}.png'.format( self.expdate, self.expnum, dname))

        self.df_rois = df_rois

    def finetune(self, rec_id, offset=[0,0], angle_adjust=0, confirm=False, savefig=False, 
        save_to='./output/rois_on_trace/'):

        import matplotlib as mpl
        
        save_to = save_to + '{}-{}/'.format(self.expdate, self.expnum)

        ids_sub = self.df_rois[self.df_rois['recording_id'] == rec_id]['recording_center'].index.tolist()

        noise_h5_paths = self.data_paths['noise_h5_paths']
        noise_h5_path = noise_h5_paths[rec_id-1]

        (stack_soma_cx, stack_soma_cy, stack_soma_cz) = self.soma / self.pixel_sizes_stack
        linestack = self.linestack
        linestack_xy = linestack.mean(2)
            
        dname = noise_h5_path.split('/')[-1].split('_')[2]
        
        logging.info('  Reading {}'.format(dname))
        
        d = load_h5_data(noise_h5_path)
        d_rec = resize_rec(d, self.data_stack)
        d_rec_rot, (origin_shift_x, origin_shift_y) = rotate_rec(d, self.data_stack)
        d_rois_rot, roi_coords_rot = rotate_roi(d, self.data_stack, angle_adjust=angle_adjust)
        
        d_rel_cy, d_rel_cx, _ = rel_position_um(self.data_soma_noise, d) / self.pixel_sizes_stack[0]
        d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 
        padding = int(max(d_rec_rot.shape)) 

        crop = linestack_xy[np.maximum(0, d_stack_cx-padding):np.minimum(d_stack_cx+padding, linestack_xy.shape[0]-1), 
                            np.maximum(0, d_stack_cy-padding):np.minimum(d_stack_cy+padding, linestack_xy.shape[0]-1)]

        d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot)
        d_rec_rot_x0 += offset[0]
        d_rec_rot_y0 += offset[1]

        roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
        d_rois_rot_crop = np.pad(d_rois_rot, pad_width=((d_rec_rot_x0, 0), (d_rec_rot_y0, 0)), mode='constant', constant_values=255)
        d_rois_rot_crop = np.ma.masked_where(d_rois_rot_crop == 255, d_rois_rot_crop)

        rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

        roi_coords_stack_xy = roi_coords_crop + np.array([np.maximum(0, d_stack_cx-padding), 
                                                              np.maximum(0, d_stack_cy-padding)])
        d_rois_rot_stack_xy = np.pad(d_rois_rot_crop, pad_width=((np.maximum(0, d_stack_cx-padding), 0), 
                                                                     (np.maximum(0, d_stack_cy-padding), 0)), 
                                                          mode='constant', constant_values=255)
        d_rois_rot_stack_xy = np.ma.masked_where(d_rois_rot_stack_xy == 255, d_rois_rot_stack_xy)

        rec_center_stack_xy = rec_center_crop + np.array([np.maximum(0,d_stack_cx-padding), 
                                                              np.maximum(0, d_stack_cy-padding)])


        ###################################
        ## Plot and check ROIs on traces ##
        ################################### 

        plt.figure(figsize=(32*3/5,32))

        ax1 = plt.subplot2grid((5,3), (0,0), rowspan=2, colspan=1)
        ax2 = plt.subplot2grid((5,3), (0,1), rowspan=2, colspan=2)
        ax3 = plt.subplot2grid((5,3), (2,0), rowspan=3, colspan=3)

        # recording region
        ax1.imshow(d_rec_rot, origin='lower', cmap=plt.cm.binary)
        ax1.imshow(d_rois_rot, origin='lower', cmap=plt.cm.viridis)
        ax1.grid('off')
        ax1.scatter(roi_coords_rot[:, 1], roi_coords_rot[:, 0], color='orange', s=80)
        for point_id, point in enumerate(roi_coords_rot):
            ax1.annotate(point_id+1, xy=point[::-1], xytext=point[::-1]-np.array([0, 2]), color='red', size=20)

        ax1.set_title('Recording Region', fontsize=24)

        # crop region
        ax2.imshow(crop, origin='lower', cmap=plt.cm.binary)
        h_d_rec_rot, w_d_rec_rot = d_rec.shape
        rect_d_rec_rot = mpl.patches.Rectangle((d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x), w_d_rec_rot, h_d_rec_rot , edgecolor='r', facecolor='none', linewidth=2)
        tmp2 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x, -d['wParamsNum'][31]) + ax2.transData
        rect_d_rec_rot.set_transform(tmp2)
        ax2.add_patch(rect_d_rec_rot)
        ax2.imshow(d_rois_rot_crop, origin='lower', cmap=plt.cm.viridis)
        ax2.scatter(roi_coords_crop[:, 1], roi_coords_crop[:, 0], s=80, color='orange')
        ax2.set_title('Cropped Region', fontsize=24)
        ax2.grid('off')

        # whole region
        ax3.imshow(linestack.mean(2), origin='lower', cmap=plt.cm.binary)
        ax3.scatter(self.soma[1]/self.pixel_sizes_stack[1], self.soma[0]/self.pixel_sizes_stack[0], s=120, marker='x')
        hd, wd = crop.shape
        rect_crop = mpl.patches.Rectangle((d_stack_cy-padding, d_stack_cx-padding), wd, hd, edgecolor='r', facecolor='none', linewidth=2)

        h_d_rec_rot, w_d_rec_rot = d_rec.shape
        rect_crop_d_rec = mpl.patches.Rectangle((d_rec_rot_y0 + d_stack_cy-padding+origin_shift_y, d_rec_rot_x0 + d_stack_cx-padding+origin_shift_x), w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
        tmp3 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+ d_stack_cy-padding+origin_shift_y, d_rec_rot_x0+d_stack_cx-padding+origin_shift_x, -d['wParamsNum'][31]) + ax3.transData
        rect_crop_d_rec.set_transform(tmp3)

        ax3.add_patch(rect_crop_d_rec)
        ax3.add_patch(rect_crop)
        ax3.imshow(d_rois_rot_stack_xy, origin='lower', cmap=plt.cm.viridis)
        ax3.scatter(roi_coords_crop[:, 1]+d_stack_cy-padding, roi_coords_crop[:, 0]+d_stack_cx-padding, s=40, color='orange')
        ax3.annotate(dname, xy=(d_rec_rot_y0 + d_stack_cy-padding-10, d_rec_rot_x0 + d_stack_cx-padding-10), color='white')
        ax3.set_title('ROIs on Cell Morpholoy', fontsize=24)
        ax3.grid('off')
        ax3.set_xlim(0,512)
        ax3.set_ylim(0,512)

        scalebar = ScaleBar(self.pixel_sizes_stack[0], units='um', location='lower left', box_alpha=0, pad=4)
        ax3.add_artist(scalebar)

        plt.suptitle('{}-{}: {}'.format(self.expdate, self.expnum, dname), fontsize=28)
        
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        if savefig:
            plt.savefig(save_to + '{}-{}-{}.png'.format( self.expdate, self.expnum, dname))

        
        if not confirm:
            logging.info('  If this looks right, set `confirm` to True to overwrite the previous ROI coordinates.')
        else:
            logging.info('  The ROI coordinates have been adjusted!')

            d_coords_xy = np.round(roi_coords_stack_xy).astype(int)

            dict_coords = {}
            dict_rec_center = {}

            for i, idx in enumerate(ids_sub):
                dict_coords[idx] = d_coords_xy[i]
                dict_rec_center[idx] = rec_center_stack_xy
        
            self.df_rois.loc[ids_sub, 'recording_center'] = dict_rec_center
            self.df_rois.loc[ids_sub, 'roi_pos_stack_xy'] = dict_coords 
            

    def calibrate(self):

        # Find ROI position on pixel coordinate stack (x, y, z)
        self.df_rois = self.df_rois.assign(
                roi_pos_stack=self.df_rois.roi_pos_stack_xy.apply(
                lambda x: calibrate_one_roi(x, self.linestack)))
        
        # Find the path each ROI on
        self.df_rois = self.df_rois.assign(
                path_id=self.df_rois.roi_pos_stack.apply(
                lambda x: on_which_path(self.df_paths, x)))
        
        # Find the location of each ROI on its coresponding path
        self.df_rois = self.df_rois.assign(
                loc_on_path=self.df_rois.roi_pos_stack.apply(
                lambda x: get_loc_on_path_stack(self.df_paths, x)))
        
        # Find ROI pos in real length coordinate. Avoid converting by voxel size due to unavoidable rounding.
        self.df_rois = self.df_rois.assign(roi_pos=pd.Series(
            {
                row[0]: self.df_paths.loc[row[1]['path_id']].path[row[1]['loc_on_path']] 
                for row in self.df_rois.iterrows()
            }
            )
        )

        # Get dendritic distance from ROI to soma
        self.df_rois = self.df_rois.assign(dendritic_distance_to_soma=pd.Series(
            { 
                row[0]: get_dendritic_distance_to_soma(
                                            self.df_paths, 
                                            row[1]['path_id'],
                                            row[1]['loc_on_path']
                                            )
                for row in self.df_rois.iterrows()
            }
            )
        )

        # Get euclidean distance from ROI to soma
        self.df_rois = self.df_rois.assign(
                euclidean_distance_to_soma=self.df_rois.roi_pos.apply(
                lambda x: get_euclidean_distance_to_soma(x, self.soma)))

        # Get number of branchpoints from ROI to soma
        self.df_rois = self.df_rois.assign(
                branches_to_soma=self.df_rois.path_id.apply(
                    lambda x: self.df_paths.loc[x]['back_to_soma']))
        self.df_rois = self.df_rois.assign(
                branches_to_soma=self.df_rois.branches_to_soma.apply(
                    lambda x: np.array(x).astype(int)
                    ))

        # # get df_rois_sub
        df_rois_sub = self.df_rois[['recording_id', 'roi_id', 'recording_center',
                                         'roi_pos', 'dendritic_distance_to_soma',
                                         'euclidean_distance_to_soma', 'branches_to_soma']]
        (stack_soma_cx, stack_soma_cy, stack_soma_cz) = self.soma / self.pixel_sizes_stack
        df_soma = pd.DataFrame(columns=(df_rois_sub.columns))
        df_soma.loc[0] = [0, 0, [stack_soma_cx, stack_soma_cy], self.soma.tolist(), 0, 0, 0]

        self.df_rois_sub = pd.concat([df_soma, df_rois_sub])
        self._get_df_data()

        # return self.df_data
        

    def get_df_sta(self, rf_pixel_size=30, rf_shape=[15,20], sigma=0.6):
        
        logging.info('  Calculating STA.')

        def upsample_rf(rf, rf_pixel_size):
            return sp.misc.imresize(rf, size=float(rf_pixel_size), interp='bilinear', mode='F')


        noise_columns = ['rec_id', 'roi_id' ,'rf_s', 'Tracetimes0_noise',
               'Triggertimes_noise','Traces0_raw_noise','Traces0_znorm_noise']

        df_sta = pd.DataFrame()
        df_sta['rec_id'] = self.df_data['rec_id']
        df_sta['roi_id'] = self.df_data['roi_id']

        if self.flip_rf:
            logging.info('  RF data acquired from Setup 2: need to be flipped.\n')
            df_sta['rf_igor'] = self.df_data[noise_columns].apply(
                lambda x: get_sta(*x, stimulus=self.stimulus_noise, source='igor'), axis=1).apply(lambda x:np.fliplr(x[0]))
            df_sta['rf_raw'] = self.df_data[noise_columns].apply(
                lambda x: get_sta(*x, stimulus=self.stimulus_noise, source='traces_raw'), axis=1).apply(lambda x:np.fliplr(x[0]))
        else:
            logging.info('  RF data acquired from Setup 3.')
            df_sta['rf_igor'] = self.df_data[noise_columns].apply(
                lambda x: get_sta(*x, stimulus=self.stimulus_noise, source='igor'), axis=1).apply(lambda x:x[0])
            df_sta['rf_raw'] = self.df_data[noise_columns].apply(
                lambda x: get_sta(*x, stimulus=self.stimulus_noise, source='traces_raw'), axis=1).apply(lambda x:x[0])            

        # get gaussian fit and smoothed rfs
        rf_labels = [name for name in df_sta.columns if 'rf' in name]
        for rf_label in rf_labels:
            df_sta['{}_gaussian'.format(rf_label)] = df_sta[rf_label].apply(
                lambda x: gaussian_fit(x))
            df_sta['{}_smoothed'.format(rf_label)] = df_sta[rf_label].apply(
                lambda x: smooth_rf(x, sigma))       

        # get upsampled rfs from smoothed rfs to real length (um)
        rf_labels = [name for name in df_sta.columns if '_smoothed' in name]
        for rf_label in rf_labels:
            df_sta['{}_upsampled'.format(rf_label)] = df_sta[rf_label].apply(
                lambda x: upsample_rf(x, rf_pixel_size))

        # get cntr and rf size for all rfs
        rf_labels = [name for name in df_sta.columns if name.startswith('rf')]
        for rf_label in rf_labels:
            df_sta[['{}_cntr'.format(rf_label), '{}_size'.format(rf_label)]] = df_sta[rf_label].apply(lambda x: get_contour(x))

        # get cntr on real size for all rfs
        rf_cntr_labels = [name for name in df_sta.columns if name.endswith('cntr')]
        for rf_cntr_label in rf_cntr_labels:
            if 'upsampled' in rf_cntr_label:
                df_sta[rf_cntr_label[:-4] + 'real_cntr'] = df_sta[rf_cntr_label] 
                df_sta[rf_cntr_label] = df_sta[rf_cntr_label] / rf_pixel_size - 0.5
            else:
                df_sta[rf_cntr_label[:-4] + 'real_cntr'] = df_sta[rf_cntr_label] * rf_pixel_size
                
        # set rf size unit to um^2
        rf_size_labels = [name for name in df_sta.columns if name.endswith('size')]
        for rf_size_label in rf_size_labels:
            if 'upsampled' in rf_size_label:
                df_sta[rf_size_label] = df_sta[rf_size_label].apply(lambda x: x /1000)
            else:
                df_sta[rf_size_label] = df_sta[rf_size_label].apply(lambda x: x * rf_pixel_size ** 2/1000)

                
        # get cntr on tree
        rfcenter =np.array(rf_shape) * int(rf_pixel_size) * 0.5
        padding = self.df_rois_sub.recording_center.apply(
            lambda x: (rfcenter-np.array(x) * self.pixel_sizes_stack[0]).astype(int)
        )
        
        rf_tree_cntr_labels = [name for name in df_sta.columns if name.endswith('real_cntr')]        
        for rf_tree_cntr_label in rf_tree_cntr_labels:
            df_sta[rf_tree_cntr_label[:-9]+'tree_cntr'] = df_sta[rf_tree_cntr_label] - padding
        
        # Initialize cntr_quality
        df_sta['cntr_quality'] = df_sta['rf_raw_smoothed_upsampled_cntr'].apply(lambda x : (x[0] == x[-1]).all())

        self.df_sta = df_sta

    def set_rf_quality(self, roi_to_remove=[], cntrtype='rf_raw_smoothed_upsampled_cntr'):

        self.df_sta['cntr_quality'] = self.df_sta[cntrtype].apply(lambda x : (x[0] == x[-1]).all())
        
        for roi_id in roi_to_remove:
             # quality[roi_id] = False
             self.df_sta.at[roi_id, 'cntr_quality'] = False

    def calibrate_cntr_offset(self, rftype='rf_raw_smoothed_upsampled'):

        # calibrate cntr on tree by soma offset

        df_sta = self.df_sta
        df_rois = self.df_rois_sub
        quality = df_sta['cntr_quality']
        all_cntrs = df_sta[rftype+'_tree_cntr']
        all_cntrs_center = np.vstack(all_cntrs.apply(lambda x: np.mean(x, 0)))[1:]
        
        rois_pos = np.vstack(df_rois.roi_pos)[:, :2][1:]

        density_center = self.density_center
      
        if quality[0] == True:
            soma_geocenter = all_cntrs[0].mean(0)
            soma_offset = density_center - soma_geocenter 
        else:
            soma_offset = np.array([0,0])
        
        rois_offsets = all_cntrs_center - rois_pos
        rois_offset = rois_offsets.mean(0)
        
        cntrs_calibrate_to_soma = all_cntrs.apply(lambda x: x-soma_offset)
        cntrs_calibrate_to_rois = all_cntrs.apply(lambda x: x-rois_offset)

        self.df_sta['cntrs_calibrate_to_soma'] = cntrs_calibrate_to_soma
        self.df_sta['cntrs_calibrate_to_rois'] = cntrs_calibrate_to_rois
        
        self.df_sta['cntrs_offset_calibrate_to_soma'] = df_sta['cntrs_calibrate_to_soma'].apply(lambda x: x.mean(0)) - df_rois.roi_pos.apply(lambda x:x[:2])
        self.df_sta['cntrs_offset_calibrate_to_rois'] = df_sta['cntrs_calibrate_to_rois'].apply(lambda x: x.mean(0)) - df_rois.roi_pos.apply(lambda x:x[:2])

    def pairwise(self, rftype='rf_raw_smoothed_upsampled'):
        
        from itertools import combinations
        
        cntr_quality = self.df_sta.cntr_quality.loc[1:] # exclude soma
        df_rois = self.df_rois_sub.loc[1:].loc[cntr_quality]
        df_sta = self.df_sta.loc[1:].loc[cntr_quality]
        df_paths = self.df_paths
        soma = self.soma
        
        total_num_pairs = np.sum(np.arange(len(df_rois)))
        logging.info('  {} pairs of ROIs are being processing.\n'.format(total_num_pairs))
        
        pair_ids = combinations(df_sta.index, 2)
        
        column_labels = ('pair_id', 'euclidian_distance_between_rois', 'dendritic_distance_between_rois',
                         'euclidian_distance_to_soma_sum', 'dendritic_distance_to_soma_sum',
                         'cbpt_angle_between_rois_deg', 'soma_angle_between_rois_deg',
                         'overlap_cntr',  'smaller_rf_size','bigger_rf_size','overlap_rf_size','overlap_index')
        df_pairs = pd.DataFrame(columns=column_labels)
        
        for pair_row_id, (roi_0, roi_1) in enumerate(pair_ids):
            
            # logging.info('  {}: {} {}'.format(pair_row_id, roi_0, roi_1))
            if pair_row_id % int(total_num_pairs / 10) == 0:
                logging.info(' ({:04d}/{:04d}) Processing pair ({} {})...'.format(pair_row_id, total_num_pairs,roi_0, roi_1))            
            
            roi_0_pos = df_rois.loc[roi_0].roi_pos
            roi_1_pos = df_rois.loc[roi_1].roi_pos
            
            roi_0_branches = set(df_rois.loc[roi_0].branches_to_soma)
            roi_1_branches = set(df_rois.loc[roi_1].branches_to_soma)
            
            roi_0_dend_dist = df_rois.loc[roi_0].dendritic_distance_to_soma
            roi_1_dend_dist = df_rois.loc[roi_1].dendritic_distance_to_soma

            roi_0_eucl_dist = df_rois.loc[roi_0].euclidean_distance_to_soma
            roi_1_eucl_dist = df_rois.loc[roi_1].euclidean_distance_to_soma
            
            roi_0_cntr = df_sta.loc[roi_0][rftype + '_tree_cntr']
            roi_1_cntr = df_sta.loc[roi_1][rftype + '_tree_cntr']
            
            # paths interection and nonoverlap
            interection = roi_0_branches & roi_1_branches
            nonintercet = roi_0_branches ^ roi_1_branches

            dist_overlap = np.sum(df_paths.loc[interection].real_length)
            
            # dendritic distance between rois
            if roi_0_branches <= roi_1_branches or roi_1_branches <= roi_0_branches:
                
                dendritic_distance_btw = abs(roi_0_dend_dist - roi_1_dend_dist)

                list_branches = [roi_0_branches,roi_1_branches]
                shorter = list_branches[np.argmin(list(map(len, list_branches)))]
                cbpt = df_paths.loc[np.argmax(df_paths.loc[shorter].back_to_soma.apply(len))].path[-1]
                cbpt_angle = angle_btw_node(roi_0_pos, roi_1_pos, cbpt)
                
                logging.debug('  set 0: {}'.format(roi_0_branches))
                logging.debug('  set 1: {}'.format(roi_1_branches))
                logging.debug('  Subsets: True')
                logging.debug('     {}'.format(roi_0_branches & roi_1_branches))
                logging.debug('     roi_0 dist: {}'.format(roi_0_dend_dist))
                logging.debug('     roi_1 dist: {}'.format(roi_1_dend_dist))
                logging.debug('       dist between: {}\n'.format(dendritic_distance_btw))
            else:
                
                dendritic_distance_btw = roi_0_dend_dist + roi_1_dend_dist - 2*dist_overlap

                if len(interection)>0:
                    cbpt = df_paths.loc[np.argmax(df_paths.loc[interection].back_to_soma.apply(len))].path[-1]
                else:
                    cbpt = soma
                    
                cbpt_angle = angle_btw_node(roi_0_pos, roi_1_pos, cbpt)
                            
                logging.debug("   roi_0 and roi_1 has nonoverlap paths")
                logging.debug("     interected paths: {}".format(interection))
                logging.debug("     not interecteded:  {}".format(nonintercet))
                logging.debug('       roi_0 dist: {}'.format(roi_0_dend_dist))
                logging.debug('       roi_1 dist: {}'.format(roi_1_dend_dist))
                logging.debug('       overlap: {}'.format(dist_overlap))
                logging.debug('       dist between: {}\n'.format(roi_0_dend_dist + roi_1_dend_dist - 2*dist_overlap))


            # euclidean distance bwetween rois

            euclidean_distance_btw = np.linalg.norm(roi_0_pos - roi_1_pos)

            # sum euclidian distance to soma 
            euclidean_distance_to_soma = roi_0_eucl_dist + roi_1_eucl_dist
            
            # sum dendritic distance to soma
            dendritic_distance_to_soma = roi_0_dend_dist + roi_1_dend_dist
            
            # angle between via soma
            soma_angle = angle_btw_node(roi_0_pos, roi_1_pos, soma)
        
            # rf overlap
            
            inner_cntr_list, sCntr_area, bCntr_area, overlap_area, overlap_index = get_cntr_interception(roi_0_cntr, roi_1_cntr)
            # store restults to dataframe
            df_pairs.loc[pair_row_id] = [(roi_0, roi_1), euclidean_distance_btw, dendritic_distance_btw,
                                     euclidean_distance_to_soma, dendritic_distance_to_soma,
                                    cbpt_angle, soma_angle,
                                    inner_cntr_list, sCntr_area, bCntr_area, overlap_area, overlap_index]
           
            logging.debug('  dendritic dist: {}'.format(dendritic_distance_btw))
            logging.debug('  euclidean dist: {}'.format(euclidean_distance_btw))
            logging.debug('  angle (soma): {}\n'.format(soma_angle))
            logging.debug('  overlap index: {}\n'.format(overlap_index))

        logging.info('  Done.\n')

        self.df_pairs = df_pairs



    ######################
    ## Plotting methods ##
    ######################

    def plot_rois(self, roi_max_distance=300):

        fig = plt.figure(figsize=(8.27,8.27))

        ax1 = plt.subplot2grid((4,4), (0,1), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=1)
        ax3 = plt.subplot2grid((4,4), (3,1), rowspan=1, colspan=3)
        ax4 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=1)
        
        soma_pos = self.soma
        dendrites = self.df_paths[self.df_paths.type == 3]   
        
        for row in dendrites.iterrows():

            path_id = row[0]
            path = row[1]['path']
            ax1.plot(path[:, 1], path[:, 0], color='black')
            ax2.plot(path[:, 2], path[:, 0], color='black')
            ax3.plot(path[:, 1], path[:, 2], color='black')
        
             

        rois_pos = np.vstack(self.df_rois.roi_pos)
        rois_dis = self.df_rois.dendritic_distance_to_soma.values
        
        ax1.scatter(soma_pos[1], soma_pos[0], c='grey', s=160, zorder=10)
        sc = ax1.scatter(rois_pos[:, 1], rois_pos[:, 0], c=rois_dis, s=40, 
                         cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)

        cbar = plt.colorbar(sc, ax=ax1, fraction=0.02, pad=.01 )
        cbar.outline.set_visible(False)

        ax2.scatter(rois_pos[:, 2], rois_pos[:, 0], c=rois_dis, s=40 * 0.8, 
                    cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)
        ax2.scatter(soma_pos[2], soma_pos[0], c='grey', s=160, zorder=10)
        
        ax3.scatter(rois_pos[:, 1], rois_pos[:, 2], c=rois_dis, s=40 * 0.8, 
                    cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)
        ax3.scatter(soma_pos[1], soma_pos[2], c='grey', s=160, zorder=10)
        
        maxlim0, _, maxlim1 = self.stack_shape * self.pixel_sizes_stack
        
        ax1.set_xlim(0, maxlim0)
        ax1.set_ylim(0, maxlim0)
        
        ax2.set_xlim(0, maxlim1)
        ax2.set_ylim(0, maxlim0)
        
        ax3.set_xlim(0, maxlim0)
        ax3.set_ylim(0, maxlim1)

        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

        ax1.axis('off')
        scalebar = ScaleBar(1, units='um', location='lower left', box_alpha=0, pad=4)
        ax1.add_artist(scalebar)

        plt.suptitle('ROIs on morph')

        return fig, [ax1, ax2, ax3, ax4]

    def plot_profile(self):

        fig = plt.figure(figsize=(8.27,8.27))

        ax1 = plt.subplot2grid((4,4), (0,1), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=1)
        ax3 = plt.subplot2grid((4,4), (3,1), rowspan=1, colspan=3)
        ax4 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=1)

        linestack_xy = np.nan_to_num(self.data_stack['Line_Stack_warped']).sum(2)
        linestack_xy[linestack_xy != 0] = 1

        linestack_xz = self.data_stack['line_stratification_xz']
        linestack_yz = self.data_stack['line_stratification_yz']

        soma_centroid = self.soma / self.pixel_sizes_stack

        sumLine = self.data_stack['sumLine']

        sumLine /= sumLine.max()
        num_layers = len(sumLine)

        ON = np.where(self.data_stack['scaledIPLdepth'] == 0)[0][0]
        OFF = np.where(self.data_stack['scaledIPLdepth'] == 1)[0][0]
        layerON  = (OFF - ON) * 0.48 + ON
        layerOFF =  (OFF - ON) * 0.77 + ON

        ax2.plot(np.arange(num_layers), sumLine * 30, color='black')  
        ax2.axvline(layerON, color='red', linestyle='dashed')
        ax2.axvline(layerOFF, color='red', linestyle='dashed')    
        ax2.annotate('ON', xy=(layerON, 0), xytext=(layerON-10, -10), zorder=10,weight="bold")
        ax2.annotate('OFF', xy=(layerOFF, 0), xytext=(layerOFF-10, -10),zorder=10, weight="bold")

        ax3.plot(sumLine * 30, np.arange(num_layers), color='black')
        ax3.axhline(layerON, color='red', linestyle='dashed')
        ax3.axhline(layerOFF, color='red', linestyle='dashed')
        ax3.annotate('ON', xy=(0, layerON), xytext=(-22, layerON-5), zorder=10,weight="bold")
        ax3.annotate('OFF', xy=(0, layerOFF), xytext=(-22, layerOFF-5),zorder=10, weight="bold")

        ax2.imshow(linestack_xz, origin='lower', cmap=plt.cm.binary)
        ax2.scatter(soma_centroid[2], soma_centroid[0], color='grey', s=120)

        ax2.axis('off')

        # sideview (left to right)
        ax3.imshow(linestack_yz.T, origin='lower', cmap=plt.cm.binary)
        ax3.scatter(soma_centroid[1], soma_centroid[2], color='grey', s=120)
        ax3.axis('off')

        # empty box
        ax4.axis('off')

        # topview

        ax1.imshow(linestack_xy, origin='lower', cmap=plt.cm.binary)
        ax1.scatter(soma_centroid[1], soma_centroid[0], color='grey', s=120)

        ax1.axis('off')
        scalebar = ScaleBar(self.pixel_sizes_stack[0], units='um', location='lower left', box_alpha=0, pad=4)
        ax1.add_artist(scalebar)

        return fig, [ax1, ax2, ax3, ax4]

    def plot_rfs(self, rftype='rf_igor_smoothed_upsampled', 
                    cntrtype='rf_igor_smoothed_upsampled_real_cntr', 
                    save_pdf=False, save_to='./output/PDF/'):

        fig_list = _plot_rfs(self.df_sta, rftype=rftype, cntrtype=cntrtype)

        return fig_list


    def plot_all_rfs(self, save_pdf=False, save_to='./output/PDF/'):

        rftype_labels = sorted([name for name in self.df_sta.columns 
                                    if name.startswith('rf') 
                                        and not name.endswith('cntr') 
                                        and not name.endswith('size') 
                                        ])
        
        figs_container = []

        for rftype in rftype_labels:
            if 'upsampled' in rftype:
                fig = _plot_rfs(self.df_sta, rftype=rftype, cntrtype=rftype+'_tree_cntr')
            else:
                fig = _plot_rfs(self.df_sta, rftype=rftype, cntrtype=rftype+'_cntr')
            figs_container.append(fig)

        if save_pdf:
            
            from matplotlib.backends.backend_pdf import PdfPages

            if not os.path.exists(save_to):
                os.makedirs(save_to)

            logging.info('  Saving RFs plot to {}'.format(save_to))
            with PdfPages(save_to + '{}-{}-rf.pdf'.format(self.expdate, self.expnum)) as pdf:
        #     for fig in [fig_3views, fig_rfs, fig_contour, fig_trend]:
                for figlist in figs_container:
                    for fig in figlist:
                        pdf.savefig(fig)

        return figs_container

    def plot_cntr(self,roi_max_distance=250, padding=0):


        soma_pos = self.soma
        dendrites = self.df_paths[self.df_paths.type == 3]   

        rois_pos = np.vstack(self.df_rois_sub.roi_pos)
        rois_dis = self.df_rois_sub.dendritic_distance_to_soma.values

        colors = np.vstack(plt.cm.viridis((rois_dis / roi_max_distance * 255).astype(int)))[:, :3]
        cntrs_calibrate_to_soma = self.df_sta['cntrs_calibrate_to_soma']
        cntrs_calibrate_to_rois = self.df_sta['cntrs_calibrate_to_rois']
        offsets_calibrate_to_soma = self.df_sta['cntrs_offset_calibrate_to_soma'].values
        offsets_calibrate_to_rois = self.df_sta['cntrs_offset_calibrate_to_rois'].values
        
        cntrs_all = [cntrs_calibrate_to_soma, cntrs_calibrate_to_rois]
        offsets_all = [offsets_calibrate_to_soma, offsets_calibrate_to_rois]
        titles_all = ['Contours calibrated by soma RF center\nto dendritic tree center offset\nif soma receptive exists',
                      'Coutours calibrated by \naverage Rois RF center to Rois position offset']
        
        quality = self.df_sta['cntr_quality'].values


        fig, ax = plt.subplots(1,2, figsize=(8.27,8.27*0.45))

        for ii, im in enumerate(ax):
            
            cntrs = cntrs_all[ii]
            offsets = offsets_all[ii]
            
            im.scatter(soma_pos[1], soma_pos[0], c='grey', s=180, zorder=10)
            im.scatter(soma_pos[1], soma_pos[0], c='red', marker='x', zorder=20)
        
            for row in dendrites.iterrows():

                path_id = row[0]
                path = row[1]['path']
                im.plot(path[:, 1], path[:, 0], color='black')
            

            
            for i, cntr in enumerate(cntrs):
                
                if not quality[i]: continue

                if i == 0: 
                    im.plot(cntr[:, 1], cntr[:, 0], color='black', lw=1, zorder=5)

                im.plot(cntr[:, 1], cntr[:, 0], color=colors[i])
                im.scatter(rois_pos[i, 1], rois_pos[i, 0], color=colors[i], zorder=10)
                im.arrow(rois_pos[i, 1], rois_pos[i, 0], offsets[i][1], offsets[i][0],fc='k', ec='k', head_width=5, head_length=5, zorder=15)

            max_lim = (self.stack_shape * self.pixel_sizes_stack)[0]+padding
            im.set_xlim(-padding, max_lim)
            im.set_ylim(-padding, max_lim)

            scalebar = ScaleBar(1, units='um', location='lower left', box_alpha=0, pad=0)
            im.add_artist(scalebar)

            im.set_title(titles_all[ii])
            im.axis('off')

        return fig, ax

    def plot_distance(self, xlim=300, ylim=50, p0=[1,1e-6,1],rftype='raw'):

        from scipy.optimize import curve_fit

        def f(x, A, B, C):
            return A*np.exp(-B*x)-C
        quality = self.df_sta['cntr_quality'].tolist()
        dist = self.df_rois_sub['dendritic_distance_to_soma'].values[quality].astype(float)
        size = self.df_sta['rf_' + rftype + 
                           '_smoothed_upsampled_size'].values[quality].astype(float)

        
        fig = plt.figure(figsize=(8.27,6))
        ax = plt.subplot(111)
        ax.scatter(dist, size, c='black')
        
        popt, pcov = curve_fit(f, dist, size, p0=p0) 
        xcurv = np.linspace(0, xlim-50, 1000)
        ycurv = f(xcurv, *popt)

        
        ax.plot(xcurv, ycurv, color='grey', lw=2)

        
        ax.set_xlabel('Dendritic distance from ROIs to soma', fontsize=12)
        ax.set_ylabel('Receptive field size', fontsize=12)
        
        ax.set_xlim(-15, xlim)
        ax.set_ylim(-5, ylim)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(0)
        ax.spines['top'].set_linewidth(0)

        ax.set_title('Distance vs. RF Size')

        return fig, ax

    def plot_hexbin(self):

        x0 = self.df_pairs.dendritic_distance_between_rois
        y0 = self.df_pairs.cbpt_angle_between_rois_deg

        x1 = self.df_pairs.dendritic_distance_to_soma_sum
        y1 = self.df_pairs.soma_angle_between_rois_deg

        z = self.df_pairs.overlap_index

        xlabel0 = 'Dendritic distances between ROIs (um)'
        ylabel0 = 'Angle between ROIs \nvia nearest common branch point(degree)'

        xlabel1 = 'Sum Dendritic distances to soma (um)'
        ylabel1 = 'Angle between ROIs\n via soma(degree)'

        fig, ax = plt.subplots(1,2, figsize=(8.27,8.27*0.45))

        xlim0 = int(np.ceil(x0.max() / 100.0)) * 100
        xlim1 = int(np.ceil(x1.max() / 100.0)) * 100
        ylim0 = int(np.ceil(y0.max() / 100.0)) * 100
        ylim1 = int(np.ceil(y0.max() / 100.0)) * 100
        
        xlim = np.maximum(xlim0, xlim1)
        ylim = np.maximum(ylim0, ylim1)
        
        ratiox0 = round(16 * x0.max() / xlim).astype(int)
        ratioy0 = round(10 * y0.max() / ylim).astype(int)
        
        ratiox1 = round(16 * x1.max() / xlim).astype(int)
        ratioy1 = round(10 * y1.max() / ylim).astype(int)
        
        if abs(ratiox0-ratiox1) < 2:
            ratiox0 = np.maximum(ratiox0, ratiox1)
            ratiox1 = np.maximum(ratiox0, ratiox1)
        if abs(ratioy0-ratioy1) < 2:
            ratioy0 = np.maximum(ratioy0, ratioy1)
            ratioy1 = np.maximum(ratioy0, ratioy1)
            
        gSize0=[ratiox0, ratioy0]
        gSize1=[ratiox1, ratioy1]
            
        print(gSize0, gSize1)
        
        
        ax[0].set_xlim(-20, xlim)
        ax[0].set_ylim(-10, ylim)
        im0 = ax[0].hexbin(x0, y0, C=z, gridsize=gSize0, alpha=0.8, vmin=0, vmax=1)

        ax[0].set_xlabel(xlabel0, fontsize=10)
        ax[0].set_ylabel(ylabel0, fontsize=10)

        ax[1].set_xlim(-20, xlim) # keep axes between two subplots consistent
        ax[1].set_ylim(-10, ylim)
        im1 = ax[1].hexbin(x1, y1, C=z, gridsize=gSize1, alpha=0.8, vmin=0, vmax=1)

        ax[1].set_xlabel(xlabel1, fontsize=10)
        ax[1].set_ylabel(ylabel1, fontsize=10)

        cb = fig.colorbar(im0, ax=ax, drawedges=False)
        cb.set_label("Overlap Index", fontsize=8)
        cb.outline.set_visible(False)

        for i in range(2):
            ax[i].spines['left'].set_linewidth(1.5)
            ax[i].spines['bottom'].set_linewidth(1.5)
            ax[i].spines['right'].set_linewidth(0)
            ax[i].spines['top'].set_linewidth(0)
            
        plt.suptitle('Overlap Hexbin')

        return fig, ax