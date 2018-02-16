import os
import logging

import numpy as np
import scipy as sp

import morphopy as mp

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from ._utils import *



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
        
        self.soma = m.df_paths[m.df_paths.type == 1].path[0].flatten()
        
        self.df_paths = m.df_paths.iloc[1:]
        
        path_stack = self.df_paths.path.apply(lambda x: (x / self.pixel_sizes_stack).round().astype(int))
        # self.df_paths.at[:, 'path_stack'] = path_stack
        self.df_paths = self.df_paths.assign(path_stack=pd.Series(path_stack))
        self.stack_shape = self.data_stack['Line_Stack_warped'].shape
        self.linestack = get_linestack(self.df_paths, self.stack_shape)

    def check(self):
        
        import matplotlib as mpl
        
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


            
    #         d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot) # the origin of the rotated rec region in the crop region 
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
     
        self.df_rois = df_rois
            
        return df_rois

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
                num_branch_to_soma=self.df_rois.path_id.apply(
                    lambda x: len(c.df_paths.loc[x]['back_to_soma'])))

        # store imaging data into dataframe
        self.df_rois = self.df_rois.assign(
            )

    def get_df_data(self):    
        
        recording_ids = np.unique(self.df_rois['recording_id'])
        
        rec_id_dict = {}
        roi_id_dict = {}
        
        Triggervalues_noise_dict = {}
        Triggertimes_noise_dict = {}
        Tracetimes0_noise_dict = {}
        Traces0_raw_noise_dict = {}

        Averages0_chirp_dict = {}
        Triggervalues_chirp_dict = {}
        Triggertimes_chirp_dict = {}
        Tracetimes0_chirp_dict = {}
        Traces0_raw_chirp_dict = {} 

        Averages0_lchirp_dict = {}
        Triggervalues_lchirp_dict = {}
        Triggertimes_lchirp_dict = {}
        Tracetimes0_lchirp_dict = {}
        Traces0_raw_lchirp_dict = {} 

        rf_s_dict = {}
        rf_t_dict = {}

        for rec_id in recording_ids:

            df_sub = self.df_rois[self.df_rois['recording_id'] == rec_id]

            filename = np.unique(df_sub['filename'])[0]
            d = load_h5_data(self.data_paths['imaging_data_dir'] + filename)

            logging.info(filename)
            chirp_filename = "_".join(filename.split('_')[:3]) + '_Chirp.h5'
            
            if chirp_filename.lower() in [chirpfile.lower().split('/')[-1]
                for chirpfile in self.data_paths['chirp_h5_paths']]:
                logging.info(chirp_filename)
                c = load_h5_data(self.data_paths['imaging_data_dir'] + chirp_filename)
            else:
                c = None
            
            lchirp_filename = "_".join(filename.split('_')[:3]) + '_lChirp.h5'
            
            if lchirp_filename.lower() in [lchirpfile.lower().split('/')[-1] for 
                    lchirpfile in self.data_paths['lchirp_h5_paths']]:
                logging.info(lchirp_filename)
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

                if c:
                    Averages0_chirp_dict[idx] = c['Averages0']
                    Triggervalues_chirp_dict[idx] = c['Triggervalues']
                    Triggertimes_chirp_dict[idx] = c['Triggertimes']
                    Tracetimes0_chirp_dict[idx] = c['Tracetimes0'][:, roi_id-1]
                    Traces0_raw_chirp_dict[idx] = c['Traces0_raw'][:, roi_id-1]
                if lc:
                    Averages0_lchirp_dict[idx] = lc['Averages0']
                    Triggervalues_lchirp_dict[idx] =lc['Triggervalues']
                    Triggertimes_lchirp_dict[idx] = lc['Triggertimes']
                    Tracetimes0_lchirp_dict[idx] = lc['Tracetimes0'][:, roi_id-1]
                    Traces0_raw_lchirp_dict[idx] = lc['Traces0_raw'][:, roi_id-1]
        
        if len([i for i in self.data_paths if 'soma_noise' in i]) != 0:
            logging.info('soma_noise_h5_path')
            s_n = load_h5_data(self.data_paths['soma_noise_h5_path'])
            rec_id_dict[0] = 0
            roi_id_dict[0] = 0

            rf_s_dict[0] = s_n['STRF_SVD_Space0'][:, :]
            rf_t_dict[0] = s_n['STRF_SVD_Time0'][:]      

            Triggervalues_noise_dict[0] = s_n['Triggervalues']
            Triggertimes_noise_dict[0] = s_n['Triggertimes']
            Tracetimes0_noise_dict[0] = s_n['Tracetimes0'][:]
            Traces0_raw_noise_dict[0] = s_n['Traces0_raw'][:] 
            
        if len([i for i in self.data_paths if 'soma_chirp' in i]) != 0:
            logging.info('soma_chirp_h5_path')
            s_c = load_h5_data(self.data_paths['soma_chirp_h5_path'])
            Averages0_chirp_dict[0] = s_c['Averages0']
            Triggervalues_chirp_dict[0] = s_c['Triggervalues']
            Triggertimes_chirp_dict[0] = s_c['Triggertimes']
            Tracetimes0_chirp_dict[0] = s_c['Tracetimes0'][:]
            Traces0_raw_chirp_dict[0] = s_c['Traces0_raw'][:]


        if len([i for i in self.data_paths if 'soma_lchirp' in i]) != 0:
            logging.info('soma_lchirp_h5_path')
            s_lc = load_h5_data(self.data_paths['soma_lchirp_h5_path'])
            Averages0_lchirp_dict[0] = s_lc['Averages0']
            Triggervalues_lchirp_dict[0] =s_lc['Triggervalues']
            Triggertimes_lchirp_dict[0] = s_lc['Triggertimes']
            Tracetimes0_lchirp_dict[0] = s_lc['Tracetimes0'][:]
            Traces0_raw_lchirp_dict[0] = s_lc['Traces0_raw'][:]

        self.df_data = pd.DataFrame(columns=('rec_id','roi_id','rf_s', 'rf_t', 
                'Triggervalues_noise','Triggertimes_noise',
                'Tracetimes0_noise', 'Traces0_raw_noise',
                'Averages0_chirp', 'Triggervalues_chirp',
                'Triggertimes_chirp', 'Tracetimes0_chirp','Traces0_raw_chirp',
                'Averages0_lchirp','Triggervalues_lchirp','Triggertimes_lchirp',
                'Tracetimes0_lchirp','Traces0_raw_lchirp'))        

        self.df_data['rec_id'] = pd.Series(rec_id_dict)
        self.df_data['roi_id'] = pd.Series(roi_id_dict)
        self.df_data['rf_s'] = pd.Series(rf_s_dict)
        self.df_data['rf_t'] = pd.Series(rf_t_dict)

        self.df_data['Triggervalues_noise'] = pd.Series(Triggervalues_noise_dict)
        self.df_data['Triggertimes_noise'] = pd.Series(Triggertimes_noise_dict)
        self.df_data['Tracetimes0_noise'] = pd.Series(Tracetimes0_noise_dict)
        self.df_data['Traces0_raw_noise'] = pd.Series(Traces0_raw_noise_dict)

        self.df_data['Averages0_chirp'] = pd.Series(Averages0_chirp_dict)
        self.df_data['Triggervalues_chirp'] = pd.Series(Triggervalues_chirp_dict)
        self.df_data['Triggertimes_chirp'] = pd.Series(Triggertimes_chirp_dict)
        self.df_data['Tracetimes0_chirp'] = pd.Series(Tracetimes0_chirp_dict)
        self.df_data['Traces0_raw_chirp'] = pd.Series(Traces0_raw_chirp_dict)

        self.df_data['Averages0_lchirp'] = pd.Series(Averages0_lchirp_dict)
        self.df_data['Triggervalues_lchirp'] = pd.Series(Triggervalues_lchirp_dict)
        self.df_data['Triggertimes_lchirp'] = pd.Series(Triggertimes_lchirp_dict)
        self.df_data['Tracetimes0_lchirp'] = pd.Series(Tracetimes0_lchirp_dict)
        self.df_data['Traces0_raw_lchirp'] = pd.Series(Traces0_raw_lchirp_dict)    
        
        return self.df_data
            