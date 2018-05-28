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
from ._rois import ROIs


__all__ = ['TTXs']

class TTXs(ROIs):

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
        self.data_paths = get_ttx_paths(rootdir, experimenter, expdate, expnum)
        

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
        # print(self.pixel_sizes_stack)

        # load stimulus
        logging.info('  loading noise stimulus\n')
        self.stimulus_noise = load_h5_data(expmeta['stimulus_path'] + 'noise.h5')['k'].reshape(15*20, -1).T
        logging.info('  loading chirp stimulus\n')
        self.stimulus_chirp = load_h5_data(expmeta['stimulus_path'] + 'chirp_old.h5')['chirp']

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

        for rec_id in recording_ids:

            df_sub = self.df_rois[self.df_rois['recording_id'] == rec_id]

            filename = np.unique(df_sub['filename'])[0]
            d = load_h5_data(self.data_paths['imaging_data_dir'] + filename)

            logging.debug("  {}".format(filename)) 
            
            for row in df_sub.iterrows():
                
                idx = row[0]
                roi_id = int(row[1]['roi_id'])
                
                rec_id_dict[idx] = rec_id
                roi_id_dict[idx] = roi_id               

                Triggervalues_noise_dict[idx] = d['Triggervalues']
                Triggertimes_noise_dict[idx] = d['Triggertimes']
                Tracetimes0_noise_dict[idx] = d['Tracetimes0'][:, roi_id-1]
                Traces0_raw_noise_dict[idx] = d['Traces0_raw'][:, roi_id-1] 

        soma_noise_h5_path = [self.data_paths[i] for i in self.data_paths if 'soma_noise' in i][0]
        if soma_noise_h5_path is not None:
            logging.debug('  soma_noise_h5_path')
            s_n = load_h5_data(self.data_paths['soma_noise_h5_path'])
            rec_id_dict[0] = 0
            roi_id_dict[0] = 0


            Triggervalues_noise_dict[0] = s_n['Triggervalues']
            Triggertimes_noise_dict[0] = s_n['Triggertimes']
            Tracetimes0_noise_dict[0] = s_n['Tracetimes0'][:, 0]
            Traces0_raw_noise_dict[0] = s_n['Traces0_raw'][:, 0]

        else:
            rec_id_dict[0] = 0
            roi_id_dict[0] = 0

            Triggervalues_noise_dict[0] = np.random.randn(1500)
            Triggertimes_noise_dict[0] = np.linspace(0, 300, 10084)
            Tracetimes0_noise_dict[0] = np.linspace(0, 300, 10084)
            Traces0_raw_noise_dict[0] = np.random.randn(10084)

        
        df_data = pd.DataFrame()   

        df_data['rec_id'] = pd.Series(rec_id_dict)
        df_data['roi_id'] = pd.Series(roi_id_dict)

        df_data['Triggervalues_noise'] = pd.Series(Triggervalues_noise_dict)
        df_data['Triggertimes_noise'] = pd.Series(Triggertimes_noise_dict)
        df_data['Tracetimes0_noise'] = pd.Series(Tracetimes0_noise_dict)
        df_data['Traces0_raw_noise'] = pd.Series(Traces0_raw_noise_dict)

        return df_data

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
        self.df_data = self._get_df_data()


    def get_df_sta(self, rf_pixel_size=30, rf_shape=[15,20], sigma=0.6, num_iters=200, threshold=2.5):
        

        df_data = self.df_data

        logging.info('  Calculating STA.')

        def upsample_rf(rf, rf_pixel_size):
            return sp.misc.imresize(rf, size=float(rf_pixel_size), interp='bilinear', mode='F')


        noise_columns = ['rec_id', 'roi_id', 'Tracetimes0_noise',
               'Triggertimes_noise','Traces0_raw_noise']

        df_sta = pd.DataFrame()
        df_sta['rec_id'] = df_data['rec_id']
        df_sta['roi_id'] = df_data['roi_id']

        if self.flip_rf:

            logging.info('  RF data acquired from Setup 2: need to be flipped.\n')
            rf_asd = df_data[noise_columns].apply(
                lambda x: get_sta(*x, stimulus=self.stimulus_noise, methods='asd', num_iters=num_iters), axis=1)
            df_sta['sRF_asd'] = rf_asd.apply(lambda x:np.fliplr(x[0]))
            df_sta['tRF_asd'] = rf_asd.apply(lambda x:x[1])
        
        else:
            logging.info('  RF data acquired from Setup 3.')

            rf_asd = df_data[noise_columns].apply(
                lambda x: get_sta(*x, stimulus=self.stimulus_noise, methods='asd', num_iters=num_iters), axis=1)
            df_sta['sRF_asd'] = rf_asd.apply(lambda x:x[0])
            df_sta['tRF_asd'] = rf_asd.apply(lambda x:x[1])

        # get smoothed and gaussian
        df_sta['sRF_asd_gaussian'] = df_sta['sRF_asd'].apply(
            lambda x: gaussian_fit(x))

        # upsampled RF to match the real length.
        df_sta['sRF_asd_upsampled'] = df_sta['sRF_asd'].apply(
            lambda x: upsample_rf(x, rf_pixel_size))

        # get cntr and rf size for all rfs
        rf_labels = [name for name in df_sta.columns if name.startswith('sRF')]
        for rf_label in rf_labels:
            df_sta[['{}_cntr'.format(rf_label), '{}_size'.format(rf_label)]] = df_sta[rf_label].apply(lambda x: get_contour(x, threshold=threshold))

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
        closed_end = df_sta['sRF_asd_upsampled_cntr'].apply(lambda x : (x[0] == x[-1]).all())
        df_sta['rf_quality'] = closed_end

        self.df_sta = df_sta.sort_index()