import os
import glob
import pickle
import logging

import numpy as np
import scipy as sp
import scipy.ndimage

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

        tmp_data_path = glob.glob('{}.pickle'.format('/gpfs01/berens/user/zhuang/Projects/rgc_dendrites_wip/Notebooks/AnalysisPaper/output/save_data2/{}'.format(self.expdate + '_' + self.expnum)))

        if len(tmp_data_path) == 1:
            logging.info('  loading saved data.\n')

            with open(tmp_data_path[0], 'rb') as f:
                tmp_data = pickle.load(f)

            self.data_paths = tmp_data['data_paths']
            self.data_stack = tmp_data['data_stack']
            self.pixel_sizes_stack = tmp_data['pixel_sizes_stack']
            self.stimulus_noise = tmp_data['stimulus_noise']
            self.soma = tmp_data['soma']
            self.density_center = tmp_data['density_center']
            self.stack_shape = tmp_data['stack_shape']
            self.linestack = tmp_data['linestack']
            self.df_paths = tmp_data['df_paths']
            self.df_rois = tmp_data['df_rois']
            self.df_rois_sub = tmp_data['df_rois_sub']
            self.df_data = tmp_data['df_data']
            self.flip_rf = tmp_data['flip_rf']

            try:
                self.df_rf = tmp_data['df_rf']
            except:
                self.df_rf = None

            try:
                # self.df_pairs_50 = tmp_data['df_pairs_50']
                # self.df_pairs_55 = tmp_data['df_pairs_55']
                self.df_pairs_60 = tmp_data['df_pairs_60']
                self.df_pairs_65 = tmp_data['df_pairs_65']
                self.df_pairs_70 = tmp_data['df_pairs_70']
                # self.df_pairs = tmp_data['df_pairs']
            except:
                # self.df_pairs_50 = None
                # self.df_pairs_55 = None
                self.df_pairs_60 = None
                self.df_pairs_65 = None
                self.df_pairs_70 = None     

            try:
                self.df_cntr = tmp_data['df_cntr']
            except:
                self.df_cntr = None
        else:
            self.data_paths = get_data_paths(rootdir, experimenter, expdate, expnum)
            

            stack_h5_path = self.data_paths['stack_h5_path']
            soma_noise_h5_path = self.data_paths['soma_noise_h5_path']  
            soma_chirp_h5_path = self.data_paths['soma_chirp_h5_path'] 
            soma_lchirp_h5_path = self.data_paths['soma_lchirp_h5_path']

            try:
                headerfile = self.data_paths['headerfile']

                with open(self.data_paths['headerfile']) as f:
                    setup_id = int([line for line in f.readlines() if 'string_setupID' in line][0].strip('\n').split('=')[-1])

                if setup_id == 2:
                    self.flip_rf = True
                else:
                    self.flip_rf = False
            except:
                self.flip_rf = False

            logging.info('  loading stack.h5\n')
            self.data_stack = load_h5_data(stack_h5_path)
            self.pixel_sizes_stack = get_pixel_size_stack(self.data_stack)
            # print(self.pixel_sizes_stack)

            # load stimulus
            logging.info('  loading noise stimulus\n')
            self.stimulus_noise = load_h5_data(expmeta['stimulus_path'] + '/old/noise.h5')['k'].T
            logging.info('  loading chirp stimulus\n')
            self.stimulus_chirp = load_h5_data(expmeta['stimulus_path'] + '/old/chirp_old.h5')['chirp']

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
            elif 'Line_stack_warped' in self.data_stack.keys():
                self.stack_shape = self.data_stack['Line_stack_warped'].shape
            else:
                self.stack_shape = self.data_stack['wDataCh0'].shape
            self.linestack = get_linestack(self.df_paths, self.stack_shape)

    def re_calculate_morphology(self):
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

    def save_pickle(self, save_df_rf=True, save_df_pairs=True, save_df_cntr=True):
        d = {
            'data_paths': self.data_paths,
            'data_stack': self.data_stack,
            'pixel_sizes_stack': self.pixel_sizes_stack,
            'stimulus_noise': self.stimulus_noise,
            'soma': self.soma,
            'density_center': self.density_center,
            'stack_shape': self.stack_shape,
            'linestack': self.linestack,
            'df_paths': self.df_paths,
            'df_rois': self.df_rois,
            'df_rois_sub': self.df_rois_sub,
            'df_data': self.df_data,
            'flip_rf': self.flip_rf,
        }
        if save_df_rf:
            d.update({'df_rf': self.df_rf})
        if save_df_pairs:
            # d.update({'df_pairs_50': self.df_pairs_50})
            # d.update({'df_pairs_55': self.df_pairs_55})
            d.update({'df_pairs_60': self.df_pairs_60})
            d.update({'df_pairs_65': self.df_pairs_65})
            d.update({'df_pairs_70': self.df_pairs_70})
        if save_df_cntr:
            d.update({'df_cntr': self.df_cntr})

        # save to save_data (data without cntr 55)
        with open('../output/save_data/{}.pickle'.format(self.expdate + '_' + self.expnum), 'wb') as f:
            pickle.dump(d, f)     

        # old path: save to tmp (data included cntr 55)
        # with open('../output/tmp/{}.pickle'.format(self.expdate + '_' + self.expnum), 'wb') as f:
        #     pickle.dump(d, f)     

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

        # Averages0_chirp_dict = {}
        Triggervalues_chirp_dict = {}
        Triggertimes_chirp_dict = {}
        Tracetimes0_chirp_dict = {}
        Traces0_raw_chirp_dict = {} 
        # Traces0_znorm_chirp_dict = {}
        # Snippets0_chirp_dict = {}
        # SnippetsTimes0_chirp_dict = {}

        # Averages0_lchirp_dict = {}
        Triggervalues_lchirp_dict = {}
        Triggertimes_lchirp_dict = {}
        Tracetimes0_lchirp_dict = {}
        Traces0_raw_lchirp_dict = {} 
        # Traces0_znorm_lchirp_dict = {}
        # Snippets0_lchirp_dict = {}
        # SnippetsTimes0_lchirp_dict = {}

        rf_s_dict = {}
        rf_t_dict = {}

        for rec_id in recording_ids:

            df_sub = self.df_rois[self.df_rois['recording_id'] == rec_id]

            filename = np.unique(df_sub['filename'])[0]
            d = load_h5_data(self.data_paths['imaging_data_dir'] + filename)

            logging.info("  {}".format(filename))
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
                
#                 rf_s_dict[idx] = d['STRF_SVD_Space0'][:, :, roi_id-1]
#                 rf_t_dict[idx] = d['STRF_SVD_Time0'][:, roi_id-1]                  

                Triggervalues_noise_dict[idx] = d['Triggervalues']
                Triggertimes_noise_dict[idx] = d['Triggertimes']
                Tracetimes0_noise_dict[idx] = d['Tracetimes0'][:, roi_id-1]
                Traces0_raw_noise_dict[idx] = d['Traces0_raw'][:, roi_id-1] 
                # Traces0_znorm_noise_dict[idx] = d['Traces0_znorm'][:, roi_id-1]

                if c:
                    # Averages0_chirp_dict[idx] = c['Averages0']
                    Triggervalues_chirp_dict[idx] = c['Triggervalues']
                    Triggertimes_chirp_dict[idx] = c['Triggertimes']
                    Tracetimes0_chirp_dict[idx] = c['Tracetimes0'][:, roi_id-1]
                    Traces0_raw_chirp_dict[idx] = c['Traces0_raw'][:, roi_id-1]
                    # Traces0_znorm_chirp_dict[idx] = c['Traces0_znorm'][:, roi_id-1]
                    # Snippets0_chirp_dict[idx] = c['Snippets0'][:, :, roi_id-1]
                    # SnippetsTimes0_chirp_dict[idx] = c['SnippetsTimes0'][:, :, roi_id-1]
                if lc:
                    # Averages0_lchirp_dict[idx] = lc['Averages0']
                    Triggervalues_lchirp_dict[idx] =lc['Triggervalues']
                    Triggertimes_lchirp_dict[idx] = lc['Triggertimes']
                    Tracetimes0_lchirp_dict[idx] = lc['Tracetimes0'][:, roi_id-1]
                    Traces0_raw_lchirp_dict[idx] = lc['Traces0_raw'][:, roi_id-1]
                    # Traces0_znorm_lchirp_dict[idx] = lc['Traces0_znorm'][:, roi_id-1]
                    # Snippets0_lchirp_dict[idx] = lc['Snippets0'][:, :, roi_id-1]
                    # SnippetsTimes0_chirp_dict[idx] = lc['SnippetsTimes0'][:, :, roi_id-1]
        
        soma_noise_h5_path = [self.data_paths[i] for i in self.data_paths if 'soma_noise' in i][0]
        if soma_noise_h5_path is not None:
            logging.debug('  soma_noise_h5_path')
            s_n = load_h5_data(self.data_paths['soma_noise_h5_path'])
            rec_id_dict[0] = 0
            roi_id_dict[0] = 0

#             rf_s_dict[0] = s_n['STRF_SVD_Space0'][:, :, 0]
#             rf_t_dict[0] = s_n['STRF_SVD_Time0'][:, 0]      
            # if 'NoiseArray3D' in d.keys():
                # NoiseArray3D_dict[0] = d['NoiseArray3D']

            Triggervalues_noise_dict[0] = s_n['Triggervalues']
            Triggertimes_noise_dict[0] = s_n['Triggertimes']
            Tracetimes0_noise_dict[0] = s_n['Tracetimes0'][:, 0]
            Traces0_raw_noise_dict[0] = s_n['Traces0_raw'][:, 0]
            # Traces0_znorm_noise_dict[0] = s_n['Traces0_znorm'][:, 0]
        else:
            rec_id_dict[0] = 0
            roi_id_dict[0] = 0

            rf_s_dict[0] = np.random.randn(15,20)
            rf_t_dict[0] = np.random.randn(49)

            Triggervalues_noise_dict[0] = Triggervalues_noise_dict[1] 
            Triggertimes_noise_dict[0] = Triggertimes_noise_dict[1]
            Tracetimes0_noise_dict[0] = Tracetimes0_noise_dict[1]
            Traces0_raw_noise_dict[0] = np.random.randn(len(Traces0_raw_noise_dict[1]))
            # Traces0_znorm_noise_dict[0] = np.random.randn(10084)   
        
        soma_chirp_h5_path = [self.data_paths[i] for i in self.data_paths if 'soma_chirp' in i][0]
        if soma_chirp_h5_path is not None:
            logging.debug('soma_chirp_h5_path: {}'.format(soma_chirp_h5_path))
            s_c = load_h5_data(self.data_paths['soma_chirp_h5_path'])
            # Averages0_chirp_dict[0] = s_c['Averages0']
            Triggervalues_chirp_dict[0] = s_c['Triggervalues']
            Triggertimes_chirp_dict[0] = s_c['Triggertimes']
            Tracetimes0_chirp_dict[0] = s_c['Tracetimes0'][:, 0]
            Traces0_raw_chirp_dict[0] = s_c['Traces0_raw'][:, 0]
            # Traces0_znorm_chirp_dict[0] = s_c['Traces0_znorm'][:, 0]
            # Snippets0_chirp_dict[0] = s_c['Snippets0'][:, :, 0]
            # SnippetsTimes0_chirp_dict[0] = s_c['SnippetsTimes0'][:, :, 0]

        soma_lchirp_h5_path = [self.data_paths[i] for i in self.data_paths if 'soma_lchirp' in i][0]
        if soma_lchirp_h5_path is not None:
            logging.debug('soma_lchirp_h5_path: {}'.format(soma_lchirp_h5_path))
            s_lc = load_h5_data(self.data_paths['soma_lchirp_h5_path'])
            # Averages0_lchirp_dict[0] = s_lc['Averages0']
            Triggervalues_lchirp_dict[0] =s_lc['Triggervalues']
            Triggertimes_lchirp_dict[0] = s_lc['Triggertimes']
            Tracetimes0_lchirp_dict[0] = s_lc['Tracetimes0'][:, 0]
            Traces0_raw_lchirp_dict[0] = s_lc['Traces0_raw'][:, 0]
            # Traces0_znorm_lchirp_dict[0] = s_lc['Traces0_znorm'][:, 0]
            # Snippets0_lchirp_dict[0] = s_lc['Snippets0'][:, :, 0]
            # SnippetsTimes0_chirp_dict[0] = s_lc['SnippetsTimes0'][:, :, 0]
 
        df_data = pd.DataFrame()   

        df_data['rec_id'] = pd.Series(rec_id_dict)
        df_data['roi_id'] = pd.Series(roi_id_dict)
        df_data['rf_s'] = pd.Series(rf_s_dict)
        df_data['rf_t'] = pd.Series(rf_t_dict)

        # df_data['NoiseArray3D'] = pd.Series(NoiseArray3D_dict)

        df_data['Triggervalues_noise'] = pd.Series(Triggervalues_noise_dict)
        df_data['Triggertimes_noise'] = pd.Series(Triggertimes_noise_dict)
        df_data['Tracetimes0_noise'] = pd.Series(Tracetimes0_noise_dict)
        df_data['Traces0_raw_noise'] = pd.Series(Traces0_raw_noise_dict)
        df_data['Traces0_znorm_noise'] = pd.Series(Traces0_znorm_noise_dict)

        # df_data['Averages0_chirp'] = pd.Series(Averages0_chirp_dict)
        df_data['Triggervalues_chirp'] = pd.Series(Triggervalues_chirp_dict)
        df_data['Triggertimes_chirp'] = pd.Series(Triggertimes_chirp_dict)
        df_data['Tracetimes0_chirp'] = pd.Series(Tracetimes0_chirp_dict)
        df_data['Traces0_raw_chirp'] = pd.Series(Traces0_raw_chirp_dict)
        # df_data['Snippets0_chirp'] = pd.Series(Snippets0_chirp_dict)
        # df_data['SnippetsTimes0_chirp'] = pd.Series(SnippetsTimes0_chirp_dict)
        
        # df_data['Averages0_lchirp'] = pd.Series(Averages0_lchirp_dict)
        df_data['Triggervalues_lchirp'] = pd.Series(Triggervalues_lchirp_dict)
        df_data['Triggertimes_lchirp'] = pd.Series(Triggertimes_lchirp_dict)
        df_data['Tracetimes0_lchirp'] = pd.Series(Tracetimes0_lchirp_dict)
        df_data['Traces0_raw_lchirp'] = pd.Series(Traces0_raw_lchirp_dict)
        # df_data['Snippets0_lchirp'] = pd.Series(Snippets0_lchirp_dict)
        # df_data['SnippetsTimes0_lchirp'] = pd.Series(SnippetsTimes0_lchirp_dict)   
        
        logging.debug('\n')

        return df_data

    def check(self, savefig=False, save_to='../output/rois_on_trace/'):
        
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
            
            if hasattr(self, 'data_soma_noise'):
                d_rel_cy, d_rel_cx, _ = rel_position_um(self.data_soma_noise, d) / self.pixel_sizes_stack[0]
                d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 
            else:
                d_rel_cy, d_rel_cx, _ = rel_position_um(self.data_stack, d) / self.pixel_sizes_stack[0]
                d_stack_cx, d_stack_cy = int(linestack_xy.shape[0]/2+d_rel_cx), int(linestack_xy.shape[0]/2+d_rel_cy) 
            
            padding = int(max(d_rec_rot.shape) * 1.2)
            # padding = int(max(d_rec_rot.shape) * 1.2)

            crop_x0, crop_x1 = np.maximum(0, d_stack_cx-padding), np.minimum(d_stack_cx+padding, linestack_xy.shape[0]-1)
            crop_y0, crop_y1 = np.maximum(0, d_stack_cy-padding), np.minimum(d_stack_cy+padding, linestack_xy.shape[0]-1)


            crop = linestack_xy[crop_x0:crop_x1, crop_y0:crop_y1]

            d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot)
            roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
            d_rois_rot_crop = np.pad(d_rois_rot, pad_width=((d_rec_rot_x0, 0), (d_rec_rot_y0, 0)), mode='constant', constant_values=1)
            d_rois_rot_crop = np.ma.masked_where(d_rois_rot_crop == 1, d_rois_rot_crop)

            rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

            roi_coords_stack_xy = roi_coords_crop + np.array([crop_x0, crop_y0])
            d_rois_rot_stack_xy = np.pad(d_rois_rot_crop, pad_width=((crop_x0, 0), (crop_y0, 0)), 
                                                          mode='constant', constant_values=1)
            d_rois_rot_stack_xy = np.ma.masked_where(d_rois_rot_stack_xy == 1, d_rois_rot_stack_xy)

            rec_center_stack_xy = rec_center_crop + np.array([crop_x0,crop_y0])
            
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
                # ax2.imshow(d_rois_rot_crop, origin='lower', cmap=plt.cm.viridis)
                ax2.scatter(roi_coords_crop[:, 1], roi_coords_crop[:, 0], s=80, color='orange')
                ax2.set_title('Cropped Region', fontsize=24)
                ax2.grid('off')

                # whole region
                ax3.imshow(linestack.mean(2), origin='lower', cmap=plt.cm.binary)
                ax3.scatter(self.soma[1]/self.pixel_sizes_stack[1], self.soma[0]/self.pixel_sizes_stack[0], s=120, marker='x')
                hd, wd = crop.shape
                rect_crop = mpl.patches.Rectangle((crop_y0, crop_x0), wd, hd, edgecolor='r', facecolor='none', linewidth=2)

                h_d_rec_rot, w_d_rec_rot = d_rec.shape
                rect_crop_d_rec = mpl.patches.Rectangle((d_rec_rot_y0 + crop_y0+origin_shift_y, d_rec_rot_x0 + crop_x0+origin_shift_x), w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
                tmp3 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+ crop_y0+origin_shift_y, d_rec_rot_x0+crop_x0+origin_shift_x, -d['wParamsNum'][31]) + ax3.transData
                rect_crop_d_rec.set_transform(tmp3)

                ax3.add_patch(rect_crop_d_rec)
                ax3.add_patch(rect_crop)
                # ax3.imshow(d_rois_rot_stack_xy, origin='lower', cmap=plt.cm.viridis)
                ax3.scatter(roi_coords_crop[:, 1]+crop_y0, roi_coords_crop[:, 0]+crop_x0, s=40, color='orange')
                ax3.annotate(dname, xy=(d_rec_rot_y0 + crop_y0-10, d_rec_rot_x0 + crop_x0-10), color='white')
                ax3.set_title('ROIs on Cell Morpholoy', fontsize=24)
                ax3.grid('off')
                ax3.set_xlim(0,linestack.shape[0])
                ax3.set_ylim(0,linestack.shape[0])

                scalebar = ScaleBar(self.pixel_sizes_stack[0], units='um', location='lower left', box_alpha=0, pad=4)
                ax3.add_artist(scalebar)

                plt.suptitle('{}-{}: {}'.format(self.expdate, self.expnum, dname), fontsize=28)
                
                if not os.path.exists(save_to):
                    os.makedirs(save_to)

                plt.savefig(save_to + '{}-{}-{}.png'.format( self.expdate, self.expnum, dname))
#                 plt.savefig(save_to + '{}-{}-{}.pdf'.format( self.expdate, self.expnum, dname))

        self.df_rois = df_rois

    def finetune(self, rec_id, offset=[0,0], pad_more=0, angle_adjust=0, confirm=False, savefig=False, savepdf=False,
        save_to='../output/rois_on_trace/'):

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
        d_rec_rot, (origin_shift_x, origin_shift_y) = rotate_rec(d, self.data_stack, angle_adjust=angle_adjust)
        d_rois_rot, roi_coords_rot = rotate_roi(d, self.data_stack, angle_adjust=angle_adjust)
        
        # if hasattr(self, 'data_soma_noise'):
        #     d_rel_cy, d_rel_cx, _ = rel_position_um(self.data_soma_noise, d) / self.pixel_sizes_stack[0]
        #     d_stack_cx, d_stack_cy = int(stack_soma_cx+d_rel_cx), int(stack_soma_cy+d_rel_cy) 
        # else:
        d_rel_cy, d_rel_cx, _ = rel_position_um(self.data_stack, d) / self.pixel_sizes_stack[0]
        d_stack_cx, d_stack_cy = int(linestack_xy.shape[0]/2+d_rel_cx), int(linestack_xy.shape[0]/2+d_rel_cy) 
        padding = int(max(d_rec_rot.shape)) + pad_more
        
        
        crop_x0, crop_x1 = np.maximum(0, d_stack_cx-padding), np.minimum(d_stack_cx+padding, linestack_xy.shape[0]-1)
        crop_y0, crop_y1 = np.maximum(0, d_stack_cy-padding), np.minimum(d_stack_cy+padding, linestack_xy.shape[0]-1)
                          
        
        crop = linestack_xy[crop_x0:crop_x1, crop_y0:crop_y1]

        d_rec_rot_x0, d_rec_rot_y0 = roi_matching(crop, d_rec_rot)
        d_rec_rot_x0 += offset[0]
        d_rec_rot_y0 += offset[1]

        roi_coords_crop = roi_coords_rot + np.array([d_rec_rot_x0, d_rec_rot_y0])
        d_rois_rot_crop = np.pad(d_rois_rot, pad_width=((d_rec_rot_x0, 0), (d_rec_rot_y0, 0)), mode='constant', constant_values=1)
        d_rois_rot_crop = np.ma.masked_where(d_rois_rot_crop == 1, d_rois_rot_crop)

        rec_center_crop = np.array([d_rec_rot.shape[0]/2,  d_rec_rot.shape[1]/2]) + np.array([d_rec_rot_x0, d_rec_rot_y0])

        roi_coords_stack_xy = roi_coords_crop + np.array([crop_x0, crop_y0])
        d_rois_rot_stack_xy = np.pad(d_rois_rot_crop, pad_width=((crop_x0, 0), (crop_y0, 0)), 
                                                          mode='constant', constant_values=1)
        d_rois_rot_stack_xy = np.ma.masked_where(d_rois_rot_stack_xy == 1, d_rois_rot_stack_xy)

        rec_center_stack_xy = rec_center_crop + np.array([crop_x0,crop_y0])


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
        rect_d_rec_rot = mpl.patches.Rectangle((d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x), 
                                               w_d_rec_rot, h_d_rec_rot , edgecolor='r', facecolor='none', linewidth=2)
        tmp2 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+origin_shift_y, d_rec_rot_x0+origin_shift_x, -d['wParamsNum'][31]-angle_adjust) + ax2.transData
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
        rect_crop = mpl.patches.Rectangle((crop_y0, 
                                           crop_x0), 
                                          wd, hd, edgecolor='r', facecolor='none', linewidth=2)
        
        h_d_rec_rot, w_d_rec_rot = d_rec.shape
        
        rect_crop_d_rec = mpl.patches.Rectangle((d_rec_rot_y0 + crop_y0+origin_shift_y,
                                                 d_rec_rot_x0 + crop_x0+origin_shift_x), 
                                                w_d_rec_rot, h_d_rec_rot, edgecolor='r', facecolor='none', linewidth=2)
        
        tmp3 = mpl.transforms.Affine2D().rotate_deg_around(d_rec_rot_y0+ crop_y0 +origin_shift_y, 
                                                           d_rec_rot_x0+ crop_x0 +origin_shift_x, -d['wParamsNum'][31]-angle_adjust) + ax3.transData
        rect_crop_d_rec.set_transform(tmp3)

        ax3.add_patch(rect_crop_d_rec)
        ax3.add_patch(rect_crop)
        ax3.imshow(d_rois_rot_stack_xy, origin='lower', cmap=plt.cm.viridis)
        ax3.scatter(roi_coords_crop[:, 1]+crop_y0, roi_coords_crop[:, 0]+crop_x0, s=40, color='orange')
        ax3.annotate(dname, xy=(d_rec_rot_y0 + crop_y0-10, d_rec_rot_x0 + crop_x0-10), color='white')
        ax3.set_title('ROIs on Cell Morpholoy', fontsize=24)
        ax3.grid('off')
        ax3.set_xlim(0,linestack.shape[0])
        ax3.set_ylim(0,linestack.shape[0])

        scalebar = ScaleBar(self.pixel_sizes_stack[0], units='um', location='lower left', box_alpha=0, pad=4)
        ax3.add_artist(scalebar)

        plt.suptitle('{}-{}: {}'.format(self.expdate, self.expnum, dname), fontsize=28)
        
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        if savefig:
            plt.savefig(save_to + '{}-{}-{}.png'.format( self.expdate, self.expnum, dname))
            
        if savepdf:
            plt.savefig(save_to + '{}-{}-{}.pdf'.format( self.expdate, self.expnum, dname))



        
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
                lambda x: get_euclidean_distance_to_one_point(x, self.soma)))

        # Get euclidean distance from ROI to dendritic density center
        self.df_rois = self.df_rois.assign(
                euclidean_distance_to_density_center=self.df_rois.roi_pos.apply(
                lambda x: get_euclidean_distance_to_one_point(x[:2], self.density_center)))        

        # Get number of branchpoints from ROI to soma
        self.df_rois = self.df_rois.assign(
                branches_to_soma=self.df_rois.path_id.apply(
                    lambda x: self.df_paths.loc[x]['back_to_soma']))
        
        self.df_rois = self.df_rois.assign(
                branches_to_soma=self.df_rois.branches_to_soma.apply(
                    lambda x: np.array(x).astype(int)
                    ))

        self.df_rois = self.df_rois.assign(
                num_branches_to_soma=self.df_rois.branches_to_soma.apply(len))

        # # get df_rois_sub
        df_rois_sub = self.df_rois[['recording_id', 'roi_id', 'recording_center',
                                         'roi_pos', 'roi_pos_stack', 'dendritic_distance_to_soma',
                                         'euclidean_distance_to_soma', 'euclidean_distance_to_density_center' , 'branches_to_soma', 'num_branches_to_soma']]
        (stack_soma_cx, stack_soma_cy, stack_soma_cz) = self.soma / self.pixel_sizes_stack
        df_soma = pd.DataFrame(columns=(df_rois_sub.columns))
                        #rec_id, roi_id, [recording_center],             'soma_pos',        'dd2soma', 'ed2soma', 'ed2dc', 'branch_to_soma'
        df_soma.loc[0] = [0,     0,      [stack_soma_cx, stack_soma_cy], self.soma.tolist(), (np.round(self.soma / self.pixel_sizes_stack).astype(int)).tolist(), 0,        0,         get_euclidean_distance_to_one_point(self.soma[:2], self.density_center), 0, 0]

        self.df_rois_sub = pd.concat([df_soma, df_rois_sub])
        self.df_data = self._get_df_data()

        # return self.df_data
        

    def get_df_rf(self, rf_pixel_size=30, df=[7,9,7], lambd=0.015):
        
        logging.info('  Calculating RF.')

        def upsample_rf(rf, rf_pixel_size, stack_pixel_size):
            from skimage.transform import resize
            scale_factor = rf_pixel_size/stack_pixel_size
            output_shape = (np.array([15, 20]) * scale_factor).astype(int)

            return resize(rf, output_shape=output_shape, mode='constant', order=3)


        noise_columns = ['rec_id', 'roi_id', 'Tracetimes0_noise',
               'Triggertimes_noise','Traces0_raw_noise']

        df_rf = pd.DataFrame()
        df_rf['rec_id'] = self.df_data['rec_id']
        df_rf['roi_id'] = self.df_data['roi_id']

        # testing
        df_data = self.df_data.copy()
        # end testing

        if self.flip_rf:

            logging.info('  RF data acquired from Setup 2: need to be flipped.\n')

            rf_opt = df_data[noise_columns].apply(
                lambda x: get_rf(*x, stimulus=self.stimulus_noise, df=df, lambd=lambd), axis=1)
            df_rf['w'] = rf_opt.apply(lambda x:x[2])
            df_rf['sRF'] = rf_opt.apply(lambda x:np.fliplr(x[0]))
            df_rf['tRF'] = rf_opt.apply(lambda x:x[1])
        
        else:
            logging.info('  RF data acquired from Setup 3.')
            
            rf_opt = df_data[noise_columns].apply(
                lambda x: get_rf(*x, stimulus=self.stimulus_noise, df=df, lambd=lambd), axis=1)
            df_rf['w'] = rf_opt.apply(lambda x:x[2])
            df_rf['sRF'] = rf_opt.apply(lambda x:x[0])
            df_rf['tRF'] = rf_opt.apply(lambda x:x[1])


        # upsampled RF to match the real length.

        df_rf['sRF_upsampled'] = df_rf['sRF'].apply(
            lambda x: upsample_rf(x, rf_pixel_size, self.pixel_sizes_stack[0]))

        self.df_rf = df_rf.sort_index()


    def draw_contours(self, rf_pixel_size=30, rf_shape=[15,20], levels=[60, 65, 70]):
        
        import itertools

        df_rf = self.df_rf.copy()
        df_rois = self.df_rois_sub
        df_cntr = pd.DataFrame()
        df_cntr[['rec_id', 'roi_id']] = df_rf[['rec_id', 'roi_id']]
        
        levels = np.array(levels)/100
        labels = [['sRF_upsampled_cntr_{0}'.format(int(lev * 100)), 'sRF_upsampled_cntr_size_{0}'.format(int(lev * 100)) ] for lev in levels]
        labels = list(itertools.chain(*labels))
        
        df_cntr[labels] = df_rf['sRF_upsampled'].apply(lambda x: get_contour(x, levels, 
                                                    self.pixel_sizes_stack[0], 
                                                    rf_pixel_size))

        # check cntr quality, then put cntr on morphology
        rfcenter =np.array([15,20]) * int(rf_pixel_size) * 0.5
        padding = self.df_rois_sub.recording_center.apply(
            lambda x: (rfcenter-np.array(x) * self.pixel_sizes_stack[0]).astype(int)
        )
        
        for lev in (levels*100).astype(int):
            
            print('Finished threshold {}...'.format(lev))
            
            # get quality
            # df_cntr['cntr_irregularity_{}'.format(lev)] = df_cntr['sRF_upsampled_cntr_{}'.format(lev)].apply(lambda x: get_irregular_index(x))
            df_cntr['cntr_counts_{}'.format(lev)] = df_cntr['sRF_upsampled_cntr_{}'.format(lev)].apply(lambda x: len(x))
            df_cntr['cntr_quality_{}'.format(lev)] = df_cntr['cntr_counts_{}'.format(lev)] == 1
            
            # put cntr on morphology
            res = []
            for j, roi_contours in enumerate(df_cntr['sRF_upsampled_cntr_{}'.format(lev)]):
                res.append([x * self.pixel_sizes_stack[0] - padding[j] for x in roi_contours]) 
            df_cntr['sRF_upsampled_tree_cntr_{}'.format(lev)] = pd.Series(res)    

            
            # calibrate cntr on tree
            density_center = self.density_center
            
            quality = df_cntr['cntr_quality_{}'.format(lev)].copy()
            if quality[0] == True:
                soma_geocenter = [np.mean(x, 0) for x in df_cntr['sRF_upsampled_tree_cntr_{}'.format(lev)][0]]
                soma_offset = [x - density_center for x in soma_geocenter]
            else:
                soma_offset = np.array([0,0])
                
            all_cntrs = df_cntr['sRF_upsampled_tree_cntr_{}'.format(lev)][quality]
            
            all_cntrs_center = all_cntrs.apply(lambda x: [np.mean(y,0) for y in x][0])
            rois_pos = np.vstack(df_rois.roi_pos)[:, :2][quality]

            # off set
            rois_offsets = np.vstack(all_cntrs_center) - rois_pos
            rois_offset = rois_offsets.mean(0)
            
            cntrs_calibrate_to_soma = all_cntrs.apply(lambda x: [y - soma_offset for y in x])
            cntrs_calibrate_to_rois = all_cntrs.apply(lambda x: [y - rois_offset for y in x])

            df_cntr['cntrs_calibrate_to_soma_{}'.format(lev)] = cntrs_calibrate_to_soma
            df_cntr['cntrs_calibrate_to_rois_{}'.format(lev)] = cntrs_calibrate_to_rois

            df_cntr['cntrs_offset_without_calibration_{}'.format(lev)] = all_cntrs.apply(lambda x:np.array([y.mean(0) for y in x])) - df_rois.roi_pos.apply(lambda x:x[:2]) 
            df_cntr['cntrs_offset_calibrate_to_soma_{}'.format(lev)] = df_cntr['cntrs_calibrate_to_soma_{}'.format(lev)][df_cntr['cntrs_calibrate_to_soma_{}'.format(lev)].notnull()].apply(lambda x: np.array([y.mean(0) for y in x])) - df_rois.roi_pos.apply(lambda x:x[:2])
            df_cntr['cntrs_offset_calibrate_to_rois_{}'.format(lev)] = df_cntr['cntrs_calibrate_to_rois_{}'.format(lev)][df_cntr['cntrs_calibrate_to_rois_{}'.format(lev)].notnull()].apply(lambda x: np.array([y.mean(0) for y in x])) - df_rois.roi_pos.apply(lambda x:x[:2])
            
            df_cntr['distance_from_RF_center_to_soma_without_calibration_{}'.format(lev)] = all_cntrs.apply(lambda x: np.mean([np.sqrt(np.sum((y.mean(0) - self.soma[:2])**2)) for y in x]))
            df_cntr['distance_from_RF_center_to_soma_calibrated_with_soma_offset_{}'.format(lev)] = df_cntr['cntrs_calibrate_to_soma_{}'.format(lev)][df_cntr['cntrs_calibrate_to_soma_{}'.format(lev)].notnull()].apply(lambda x: np.mean([np.sqrt(np.sum((y.mean(0) - self.soma[:2])**2)) for y in x]))
            df_cntr['distance_from_RF_center_to_soma_calibrated_with_average_offset_{}'.format(lev)] = df_cntr['cntrs_calibrate_to_rois_{}'.format(lev)][df_cntr['cntrs_calibrate_to_rois_{}'.format(lev)].notnull()].apply(lambda x: np.mean([np.sqrt(np.sum((y.mean(0) - self.soma[:2])**2)) for y in x]))

            df_cntr['distance_from_RF_center_to_ROI_without_calibration_{}'.format(lev)] = df_cntr['cntrs_offset_without_calibration_{}'.format(lev)][df_cntr['cntrs_offset_without_calibration_{}'.format(lev)].notnull()].apply(lambda x: np.mean([np.sqrt(np.sum(y**2)) for y in x]))
            df_cntr['distance_from_RF_center_to_ROI_calibrated_with_soma_offset_{}'.format(lev)] = df_cntr['cntrs_offset_calibrate_to_soma_{}'.format(lev)][df_cntr['cntrs_offset_calibrate_to_soma_{}'.format(lev)].notnull()].apply(lambda x: np.mean([np.sqrt(np.sum(y**2)) for y in x]))
            df_cntr['distance_from_RF_center_to_ROI_calibrated_with_average_offset_{}'.format(lev)] = df_cntr['cntrs_offset_calibrate_to_rois_{}'.format(lev)][df_cntr['cntrs_offset_calibrate_to_rois_{}'.format(lev)].notnull()].apply(lambda x: np.mean([np.sqrt(np.sum(y**2)) for y in x]))
            
        df_cntr['cntr_quality'] = df_cntr[[
            'cntr_quality_{}'.format(int(lev*100)) for lev in levels]].all(1)
        
        self.df_cntr = df_cntr
        self.levels = levels

    def reset_cntr_quality(self):
        """
        remove cntr_quality_55. 
        """
        self.df_cntr['cntr_quality'] = self.df_cntr[[
                      'cntr_quality_60', 
                      'cntr_quality_65', 
                      'cntr_quality_70'
                      ]].all(1)

    def set_cntr_quality_to_false(self, roi_to_remove=[]):
        for roi_id in roi_to_remove:
             # quality[roi_id] = False
             self.df_cntr.at[roi_id, 'cntr_quality'] = False

    def set_cntr_quality_to_true(self, roi_to_add=[]):
        for roi_id in roi_to_add:
             # quality[roi_id] = False
             self.df_cntr.at[roi_id, 'cntr_quality'] = True

 
    def pairwise(self):

        from itertools import combinations
        
        cntr_quality = self.df_cntr['cntr_quality'].loc[1:] # exclude soma

        # for ilev, lev in enumerate(np.arange(55, 75, 5)):
        for ilev, lev in enumerate(np.arange(60, 75, 5)):
            
            print('\nStart calculating pairwise data for contour level {}\n'.format(lev))
            
            # cntr_quality = self.df_cntr['cntr_quality_{}'.format(lev)].loc[1:] # exclude soma
            df_rois = self.df_rois_sub.loc[1:].loc[cntr_quality]
            df_rf = self.df_rf.loc[1:].loc[cntr_quality]
            df_paths = self.df_paths
            df_cntr = self.df_cntr

            soma = self.soma

            total_num_pairs = np.sum(np.arange(len(df_rois)))
            print('  {} pairs of ROIs are being processing.\n'.format(total_num_pairs))

            pair_ids = combinations(df_rois.index, 2)

            column_labels = ('pair_id', 'euclidian_distance_between_rois', 'dendritic_distance_between_rois',
                             'euclidian_distance_to_soma_sum', 'dendritic_distance_to_soma_sum',
                             'cbpt_angle_between_rois_deg', 'soma_angle_between_rois_deg',
                             'overlap_cntr','overlap_index')
            df_pairs = pd.DataFrame(columns=column_labels)

            for pair_row_id, (roi_0, roi_1) in enumerate(pair_ids):

                # logging.info('  {}: {} {}'.format(pair_row_id, roi_0, roi_1))
                every_ten = int(total_num_pairs / 10)
                if every_ten == 0:
                    print(' ({:04d}/{:04d}) Processing pair ({} {})...'.format(pair_row_id, total_num_pairs,roi_0, roi_1))     
                elif pair_row_id % every_ten == 0:
                    print(' ({:04d}/{:04d}) Processing pair ({} {})...'.format(pair_row_id, total_num_pairs,roi_0, roi_1))            

                roi_0_pos = df_rois.loc[roi_0].roi_pos
                roi_1_pos = df_rois.loc[roi_1].roi_pos

                roi_0_branches = set(df_rois.loc[roi_0].branches_to_soma)
                roi_1_branches = set(df_rois.loc[roi_1].branches_to_soma)

                roi_0_dend_dist = df_rois.loc[roi_0].dendritic_distance_to_soma
                roi_1_dend_dist = df_rois.loc[roi_1].dendritic_distance_to_soma

                roi_0_eucl_dist = df_rois.loc[roi_0].euclidean_distance_to_soma
                roi_1_eucl_dist = df_rois.loc[roi_1].euclidean_distance_to_soma

                roi_0_cntr = df_cntr.loc[roi_0]['sRF_upsampled_tree_cntr_{}'.format(lev)]
                roi_1_cntr = df_cntr.loc[roi_1]['sRF_upsampled_tree_cntr_{}'.format(lev)]

                # paths interection and nonoverlap
                interection = roi_0_branches & roi_1_branches
                nonintercet = roi_0_branches ^ roi_1_branches

                dist_overlap = np.sum(df_paths.loc[interection].real_length)

                # dendritic distance between rois
                if roi_0_branches <= roi_1_branches or roi_1_branches <= roi_0_branches:

                    dendritic_distance_btw = abs(roi_0_dend_dist - roi_1_dend_dist)

                    list_branches = [roi_0_branches,roi_1_branches]
                    shorter = list_branches[np.argmin(list(map(len, list_branches)))]
                    cbpt = df_paths.loc[df_paths.loc[shorter].back_to_soma.apply(len).idxmax()].path[-1]
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
                        cbpt = df_paths.loc[df_paths.loc[interection].back_to_soma.apply(len).idxmax()].path[-1]
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

                # inner_cntr_list, sCntr_area, bCntr_area, overlap_area, overlap_index = get_cntr_interception(roi_0_cntr, roi_1_cntr)
                inner_cntr_list, overlap_index = get_cntr_interception(roi_0_cntr, roi_1_cntr)
                # store restults to dataframe
                df_pairs.loc[pair_row_id] = [(roi_0, roi_1), euclidean_distance_btw, dendritic_distance_btw,
                                         euclidean_distance_to_soma, dendritic_distance_to_soma,
                                        cbpt_angle, soma_angle,
                                        inner_cntr_list, overlap_index]

            print('  Done.\n')

            exec('self.df_pairs_{} = df_pairs'.format(lev))

    def save_offset_df(self, roi_id, exptype):
        
        
        out0 = self.df_rf.loc[roi_id][['rec_id','roi_id', 'sRF_upsampled_size', 'distance_from_RF_center_to_ROI_without_calibration', 'distance_from_RF_center_to_soma_without_calibration', 'distance_from_RF_center_to_ROI_calibrated_with_average_offset', 'distance_from_RF_center_to_soma_calibrated_with_average_offset']]
        out1 = self.df_rois.loc[roi_id][['dendritic_distance_to_soma', 'euclidean_distance_to_soma']]
        
        o = pd.concat([out0, out1], axis=1)
        
        o.to_csv('../output/ttx/{}-{}/{}-{}-{}.csv'.format(self.expdate, self.expnum, self.expdate, self.expnum, exptype))


    ######################
    ## Plotting methods ##
    ######################

    def plot_rois(self, roi_max_distance=300, plot_rois=True, plot_morph=True, plot_stratification=False):

        def get_sumline(self):
    
            sumLine = self.data_stack['sumLine']

            sumLine /= sumLine.max()
            num_layers = len(sumLine)
                
            depth0 = np.vstack(np.vstack(self.df_paths.path) / self.pixel_sizes_stack).max(0)[2]
            depth1 = np.where([self.data_stack['line_stratification_yz'].T.mean(1) != 0])[1][-1]
                
        #     return sumLine, depth0, depth1
            new_sum = np.roll(sumLine, -int(depth1-depth0))
            
            ON = np.where(self.data_stack['scaledIPLdepth'] == 0)[0][0]
            OFF = np.where(self.data_stack['scaledIPLdepth'] == 1)[0][0]
            layerON  = ((OFF - ON) * 0.48 + ON - (depth1 - depth0)) * self.pixel_sizes_stack[2]
            layerOFF =  ((OFF - ON) * 0.77 + ON - (depth1 - depth0)) * self.pixel_sizes_stack[2]
            
            return new_sum, layerON, layerOFF

        # sumLine, layerON, layerOFF = get_sumline(self)

        fig = plt.figure(figsize=(8.27,8.27))

        ax1 = plt.subplot2grid((4,4), (0,1), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=1)
        ax3 = plt.subplot2grid((4,4), (3,1), rowspan=1, colspan=3)
        ax4 = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=1)
        
        soma_pos = self.soma
        # dendrites = self.df_paths[self.df_paths.type == 3]   
        dendrites = self.df_paths

        maxlim0, _, maxlim1 = self.stack_shape * self.pixel_sizes_stack
        # num_layers = len(sumLine)

        if plot_morph:

            for row in dendrites.iterrows():

                path_id = row[0]
                path = row[1]['path']
                ax1.plot(path[:, 0], path[:, 1], color='black')
                ax2.plot(path[:, 2], path[:, 1], color='black')
                ax3.plot(path[:, 0], path[:, 2], color='black')
            
                             
        if plot_rois:

            rois_pos = np.vstack(self.df_rois.roi_pos)
            rois_dis = self.df_rois.dendritic_distance_to_soma.values
            
            # soma
            ax1.scatter(soma_pos[0], soma_pos[1], c='grey', s=160, zorder=10)
            ax2.scatter(soma_pos[2], soma_pos[1], c='grey', s=160, zorder=10)
            ax3.scatter(soma_pos[0], soma_pos[2], c='grey', s=160, zorder=10)

            sc = ax1.scatter(rois_pos[:, 0], rois_pos[:, 1], c=rois_dis, s=40, 
                             cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)
            cbar = plt.colorbar(sc, ax=ax1, fraction=0.02, pad=.01 )
            cbar.outline.set_visible(False)

            ax2.scatter(rois_pos[:, 2], rois_pos[:, 1], c=rois_dis, s=40 * 0.8, 
                        cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)
            ax3.scatter(rois_pos[:, 0], rois_pos[:, 2], c=rois_dis, s=40 * 0.8, 
                        cmap=plt.cm.viridis, vmin=0, vmax=roi_max_distance, zorder=10)
            
            
            # ax3.plot(sumLine * 30, np.linspace(0, maxlim1 ,num_layers), color='black')
            
        if plot_stratification:

            ax3.axhline(layerON, color='red', linestyle='dashed')
            ax3.axhline(layerOFF, color='red', linestyle='dashed')    

            ax3.annotate('ON', xy=(0, layerON), xytext=(-22, layerON-5), zorder=10,weight="bold")
            ax3.annotate('OFF', xy=(0, layerOFF), xytext=(-22, layerOFF-5),zorder=10, weight="bold")
            
        
        # ax1.set_xlim(0, maxlim0)
        # ax1.set_ylim(0, maxlim0)
        
        # ax2.set_xlim(0, maxlim1)
        # ax2.set_ylim(0, maxlim0)
        
        # ax3.set_xlim(0, maxlim0)
        # ax3.set_ylim(0, maxlim1)

        ax1.set_xlim(0, 350)
        ax1.set_ylim(0, 350)
        
        ax2.set_xlim(0, maxlim1)
        ax2.set_ylim(0, 350)
        
        ax3.set_xlim(0, 350)
        ax3.set_ylim(0, maxlim1)
        
        ax1.invert_yaxis()
        ax2.invert_yaxis()
#         ax3.invert_yaxis()
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

        ax1.axis('off')
        scalebar = ScaleBar(1, units='um', location='lower left', box_alpha=0, pad=4)
        ax1.add_artist(scalebar)

        plt.suptitle('ROIs on morph')

        return fig, [ax1, ax2, ax3, ax4]


    def plot_all_rfs(self, kind='sRF_upsampled'):

        fig_list = _plot_rfs(self.df_rf, self.df_cntr, kind=kind)

        return fig_list



    def plot_cntr(self,roi_max_distance=250, padding=50):


        soma_pos = self.soma
        # dendrites = self.df_paths[self.df_paths.type == 3]   
        dendrites = self.df_paths
        levels = self.levels

        rois_pos = np.vstack(self.df_rois_sub.roi_pos)
        rois_dis = self.df_rois_sub.dendritic_distance_to_soma.values

        colors = np.vstack(plt.cm.viridis((rois_dis / roi_max_distance * 255).astype(int)))[:, :3]

        fig, ax = plt.subplots(len(levels),3, figsize=(8.27,11.69))
        ax = ax.flatten()

        quality = self.df_cntr['cntr_quality'].values

        for ilev, lev in enumerate(np.array(levels*100).astype(int)):
            
            cntrs_without_calibration = self.df_cntr['sRF_upsampled_tree_cntr_{}'.format(lev)]
            cntrs_calibrate_to_soma = self.df_cntr['cntrs_calibrate_to_soma_{}'.format(lev)]
            cntrs_calibrate_to_rois = self.df_cntr['cntrs_calibrate_to_rois_{}'.format(lev)]
            offsets_without_calibration = self.df_cntr['cntrs_offset_without_calibration_{}'.format(lev)].values
            offsets_calibrate_to_soma = self.df_cntr['cntrs_offset_calibrate_to_soma_{}'.format(lev)].values
            offsets_calibrate_to_rois = self.df_cntr['cntrs_offset_calibrate_to_rois_{}'.format(lev)].values

            cntrs_all = [cntrs_without_calibration, cntrs_calibrate_to_soma, cntrs_calibrate_to_rois]
            offsets_all = [offsets_without_calibration, offsets_calibrate_to_soma, offsets_calibrate_to_rois]
            titles_all = ['Without adjusting',
                          'Adjusted by soma offset(if exists)',
                          'Adjusted by ROIs mean offset']

            s, e = [ilev*3, ilev*3+3]
            
            for ii, im in enumerate(ax[s:e]):

                cntrs = cntrs_all[ii]
                offsets = offsets_all[ii]

                im.scatter(soma_pos[0], soma_pos[1], c='grey', s=180, zorder=10)
                im.scatter(soma_pos[0], soma_pos[1], c='red', marker='x', zorder=20)

                for row in dendrites.iterrows():

                    path_id = row[0]
                    path = row[1]['path']
                    im.plot(path[:, 0], path[:, 1], color='black')
                
                for i, cnt in enumerate(cntrs):
                
                    if quality[i] == False: 
                        continue

                    for cntr in cnt:
                        im.plot(cntr[:, 0], cntr[:, 1], color=colors[i])
                        im.scatter(rois_pos[i, 0], rois_pos[i, 1], color=colors[i], zorder=10)
                        im.arrow(rois_pos[i, 0], rois_pos[i, 1], offsets[i][0][0], offsets[i][0][1],fc='k', ec='k', head_width=5, head_length=5, zorder=15)
    #                     im.annotate(i, xy=cntr.mean(0), xytext=cntr.mean(0)-np.array([0, 2]), color='red', size=20)
                        
                max_lim = (self.stack_shape * self.pixel_sizes_stack)[0]
                max_lim = np.maximum(max_lim, 350)
                im.set_xlim(-padding, max_lim)
                im.set_ylim(-padding, max_lim)

                scalebar = ScaleBar(1, units='um', location='lower left', box_alpha=0, pad=0)
                im.add_artist(scalebar)

                im.set_title(titles_all[ii], fontsize=8)
                im.axis('off')
                im.invert_yaxis()
                im.axis('equal')

        # plt.autoscale(True)
        # plt.suptitle('contour thresholds: 0.55, 0.6, 0.65, 0.7')
        plt.suptitle('contour thresholds')
        
        return fig, ax

    def plot_distance(self, xlim=300, ylim=50):

        from scipy.optimize import curve_fit

        fig, ax = plt.subplots(1,3, figsize=(8.27,8.27 / 3))
        ax = ax.flatten()

        quality = self.df_cntr['cntr_quality'].tolist()

        # for i, lev in enumerate(np.arange(55, 75, 5)):
        for i, lev in enumerate(np.arange(60, 75, 5)):
            # quality = self.df_cntr['cntr_quality_{}'.format(lev)].tolist()
            size = np.array([x[0] for x in self.df_cntr['sRF_upsampled_cntr_size_{}'.format(lev)].values[quality]]).astype(np.float32)
            dist = self.df_rois_sub['dendritic_distance_to_soma'].values[quality].astype(float)

    #         fig = plt.figure(figsize=(8.27,6))
            ax[i].scatter(dist, size, c='black', alpha=0.6)

            if i == 1:
                ax[i].set_xlabel('Dendritic distance to soma', fontsize=12)
            
            ax[i].set_ylabel('RF size (thrd {})'.format(lev/100), fontsize=12)

            ax[i].set_xlim(-15, xlim)
            ax[i].set_ylim(-5, ylim)
            ax[i].spines['left'].set_linewidth(1.5)
            ax[i].spines['bottom'].set_linewidth(1.5)
            ax[i].spines['right'].set_linewidth(0)
            ax[i].spines['top'].set_linewidth(0)

        # ax[-1].axis('off')
        # plt.autoscale(True)
        plt.suptitle('Dendritic distance vs. RF size with diff thrds')
        return fig, ax


    def plot_hexbin(self):


        fig, ax = plt.subplots(1,3, figsize=(8.27,8.27 / 3 ))
        ax = ax.flatten()
        # for ilev, lev in enumerate(np.arange(55,75,5)):
        for ilev, lev in enumerate(np.arange(60,75,5)):
            # if lev == 50:
            #     df_pairs = self.df_pairs_50
            # if lev == 55:
            #     df_pairs = self.df_pairs_55
            if lev == 60:
                df_pairs = self.df_pairs_60
            elif lev == 65:
                df_pairs = self.df_pairs_65
            elif lev == 70:
                df_pairs = self.df_pairs_70

            x0 = df_pairs.dendritic_distance_between_rois
            y0 = df_pairs.cbpt_angle_between_rois_deg

            z = df_pairs.overlap_index

            xlabel0 = 'Dendritic distances between ROIs (um)'
            ylabel0 = 'Angle between ROIs'


            if ilev == 0:
                xlim0 = int(np.ceil(x0.max() / 100.0)) * 100
                ylim0 = int(np.ceil(y0.max() / 100.0)) * 100


                ratiox0 = round(16 * x0.max() / xlim0).astype(int)
                ratioy0 = round(10 * y0.max() / ylim0).astype(int)

                gSize0=[ratiox0, ratioy0]            

            # print(xlim0, ylim0)
            ax[ilev].set_xlim(-20, xlim0)
            ax[ilev].set_ylim(-10, ylim0)

            ax[ilev].hexbin(x0, y0, C=z, gridsize=gSize0, alpha=0.8, vmin=0, vmax=1)

    #         if lev == 70:
    #             im0 = ax[ilev].hexbin(x0, y0, C=z, gridsize=gSize0, alpha=0.8, vmin=0, vmax=1)

    #             cb = fig.colorbar(im0, ax=ax, drawedges=False)
    #             cb.set_label("Overlap Index", fontsize=8)
    #             cb.outline.set_visible(False)
    #         else:
    #             ax[ilev].hexbin(x0, y0, C=z, gridsize=gSize0, alpha=0.8, vmin=0, vmax=1)
            if ilev in [1]:
                ax[ilev].set_xlabel(xlabel0, fontsize=10)
            if ilev in [0]:
                ax[ilev].set_ylabel(ylabel0, fontsize=10)

            ax[ilev].set_title('Overlap thrd {}'.format(lev))

            ax[ilev].spines['left'].set_linewidth(1.5)
            ax[ilev].spines['bottom'].set_linewidth(1.5)
            ax[ilev].spines['right'].set_linewidth(0)
            ax[ilev].spines['top'].set_linewidth(0)


        return fig, ax


    def plot_offset(self, lim=250):

        def duplicate_by_index(l, idx):
            '''
            input: [1,2,3,4], [0, 2]
            return: [1, 1, 2, 3, 3, 4]
            '''
            import copy
            idx = np.array(idx)
            ll = list(copy.deepcopy(l))
            for ii in range(len(idx)):
                i2d = idx[ii]
                v2d = ll[i2d]
                ll.insert(i2d, v2d)
                idx += 1
            return np.array(ll)

        soma_pos = self.soma
        dendrites = self.df_paths[self.df_paths.type == 3]   
        
        fig, ax = plt.subplots(3,2, figsize=(8.27,11.69), sharex=True, sharey=True)
    #     ax = ax.flatten()
        quality = self.df_cntr['cntr_quality'].values

        # for ilev, lev in enumerate(np.arange(55, 75, 5)):
        for ilev, lev in enumerate(np.arange(60, 75, 5)):
            
            # quality = self.df_cntr['cntr_quality_{}'.format(lev)].values
            rois_pos = np.vstack(self.df_rois_sub.roi_pos)[quality]

            rois_dis0 = self.df_rois_sub.dendritic_distance_to_soma.values[quality]
            rois_dis1 = self.df_rois_sub.euclidean_distance_to_density_center.values[quality]

            offsets = self.df_cntr['cntrs_offset_calibrate_to_rois_{}'.format(lev)].values[quality]

            index_to_duplicate = np.where([len(x)==2 for x in offsets])[0]
            rois_dis_updated = duplicate_by_index(rois_dis0, index_to_duplicate)

            dist_offsets = np.sqrt((np.vstack(offsets) ** 2).sum(1))

            ax[ilev,0].scatter(rois_dis_updated, dist_offsets, color='black', s=5)
            ax[ilev,0].set_xlim(-10, lim)
            ax[ilev,0].set_ylim(-10, lim)

            ax[ilev,0].spines['left'].set_linewidth(1.5)
            ax[ilev,0].spines['bottom'].set_linewidth(1.5)
            ax[ilev,0].spines['right'].set_linewidth(0)
            ax[ilev,0].spines['top'].set_linewidth(0)
            ax[ilev,0].set_ylabel('Offset(um) cntr thrd ({})'.format(lev/100), fontsize=8)
            


            index_to_duplicate = np.where([len(x)==2 for x in offsets])[0]
            rois_dis_updated = duplicate_by_index(rois_dis1, index_to_duplicate)

            dist_offsets = np.sqrt((np.vstack(offsets) ** 2).sum(1))

            ax[ilev, 1].scatter(rois_dis_updated, dist_offsets, color='black', s=5)
        #             im.set_title(titles_all[ii], fontsize=8)
            ax[ilev, 1].set_xlim(-10, lim)
            ax[ilev, 1].set_ylim(-10, lim)


            ax[ilev, 1].spines['left'].set_linewidth(1.5)
            ax[ilev, 1].spines['bottom'].set_linewidth(1.5)
            ax[ilev, 1].spines['right'].set_linewidth(0)
            ax[ilev, 1].spines['top'].set_linewidth(0)
        #     ax[1].set_ylabel('Offset between ROI and RF center(um)', fontsize=8)
        
        ax[2,0].set_xlabel('Dendritic distance from ROI to soma', fontsize=8)
        ax[2,1].set_xlabel('Euclidean distance from ROI to dendritic density center', fontsize=8)

        plt.suptitle('offset vs distance')
        
        return fig, ax



    def save_csv(self, celltype):

        output = pd.DataFrame([self.df_rois_sub.recording_id,
                       self.df_rois_sub.roi_id,
         self.df_rois_sub.dendritic_distance_to_soma,
         self.df_rois_sub.euclidean_distance_to_density_center,
         
         self.df_cntr['sRF_upsampled_cntr_size_60'].apply(lambda x: np.sum(x)),
         self.df_cntr['cntrs_offset_calibrate_to_rois_60'].apply(lambda x: np.sqrt(np.sum(x**2))).apply(lambda x: np.nan_to_num(x)), 
         self.df_cntr['cntr_quality_60'].astype(int),
                               
         self.df_cntr['sRF_upsampled_cntr_size_65'].apply(lambda x: np.sum(x)),
         self.df_cntr['cntrs_offset_calibrate_to_rois_65'].apply(lambda x: np.sqrt(np.sum(x**2))).apply(lambda x: np.nan_to_num(x)), 
         self.df_cntr['cntr_quality_65'].astype(int),
                               
         self.df_cntr['sRF_upsampled_cntr_size_70'].apply(lambda x: np.sum(x)),
         self.df_cntr['cntrs_offset_calibrate_to_rois_70'].apply(lambda x: np.sqrt(np.sum(x**2))).apply(lambda x: np.nan_to_num(x)), 
         self.df_cntr['cntr_quality_70'].astype(int),

         self.df_cntr['cntr_quality'].astype(int),
                              ]).T

        output.to_csv('../output/csv/{}/{}_{}.csv'.format(celltype, self.expdate, self.expnum), ',')

        # for lev in np.arange(55, 75, 5):
        for lev in np.arange(60, 75, 5):
            
            # if lev == 50:
            #     df_pairs = self.df_pairs_50
            # if lev == 55:
            #     df_pairs = self.df_pairs_55
            if lev == 60:
                df_pairs = self.df_pairs_60
            elif lev == 65:
                df_pairs = self.df_pairs_65
            elif lev == 70:
                df_pairs = self.df_pairs_70
            
            output = pd.DataFrame([df_pairs.pair_id,
                df_pairs.dendritic_distance_between_rois,
                df_pairs.cbpt_angle_between_rois_deg.apply(lambda x: np.nan_to_num(x)),   
                df_pairs.dendritic_distance_to_soma_sum,
                df_pairs.soma_angle_between_rois_deg,
                df_pairs.overlap_index,
                                  ]).T

            output.to_csv('../output/csv/{}/{}_{}_pairs_{}.csv'.format(celltype, self.expdate, self.expnum, lev), ',')
