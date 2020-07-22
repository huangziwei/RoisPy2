import os
import h5py
import scipy
import logging

import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy

from skimage.transform import resize, rotate
from skimage.feature import match_template
from shapely.geometry import Polygon

def get_logger(loglevel):

    """
    Log out useful or debug infos.

    Parameters
    ---------
    loglevel: str
        'debug', 'info', 'warning', 'error', 'critical'.

    Returns
    -------
    logger: logging.RootLogger
        a logging object for turning on and off the log.

    """

    logger = logging.getLogger()

    if loglevel is not None:

        LEVELS = {'debug': logging.DEBUG,
                  'info': logging.INFO,
                  'warning': logging.WARNING,
                  'error': logging.ERROR,
                  'critical': logging.CRITICAL}

        try:
            LEVEL = LEVELS[loglevel]
            logger.setLevel(LEVEL)
        except ValueError:
            logger.setLevel(logging.INFO)
            logging.info('  Please enter a valid logging mode (DEBUG, INFO, WARNING, ERROR, CRITICAL).')
            logger.setLevel(logging.ERROR)

        return logger
    else:
        return None


def get_data_paths(rootdir, experimenter, expdate, expnum):

    exproot = rootdir + '/' + experimenter + '/' + expdate + '/' + str(expnum) + '/'
    imaging_data_dir = exproot + 'Pre/'
    morph_data_dir = exproot + 'Raw/'

    noise_h5_paths = []
    chirp_h5_paths = []
    lchirp_h5_paths = []    

    for file in os.listdir(exproot):
        if ('.ini' in file.lower()):
            headerfile_path = exproot + file

    for file in os.listdir(morph_data_dir):
        if ('.swc' in file.lower()):
            swc_path = morph_data_dir + file
    
    for file in os.listdir(imaging_data_dir):
#         logging.info(file)
        if ('stack' in file.lower()):
            stack_h5_path = imaging_data_dir + file 
        
        if ('_s_dnoise' in file.lower()):
            soma_noise_h5_path = imaging_data_dir + file

        if ('_s_chirp' in file.lower()):
            soma_chirp_h5_path = imaging_data_dir + file

        if ('_s_lchirp' in file.lower()):
            soma_lchirp_h5_path = imaging_data_dir + file
        
        if ('dnoise' in file.lower() and '_s' not in file.lower() and 'ttx' not in file.lower()):
            noise_h5_paths.append(imaging_data_dir + file)

        if ('_chirp' in file.lower() and '_s' not in file.lower() and 'ttx' not in file.lower()):
            chirp_h5_paths.append(imaging_data_dir + file)
        
        if ('_lchirp' in file.lower() and '_s' not in file.lower() and 'ttx' not in file.lower()):
            lchirp_h5_paths.append(imaging_data_dir + file)

    noise_h5_paths.sort()
    chirp_h5_paths.sort()
    lchirp_h5_paths.sort()
    
    logging.info('  Root Dir: \n\t\t{}'.format(rootdir))
    logging.info('  Imaging Data Dir: \n\t\t{}\n'.format(imaging_data_dir))
    logging.info('  Morph Data Dir: \n\t\t{}\n'.format(morph_data_dir))
    
    try:
        logging.info('  stack_h5_path: \n\t\t{} \n'.format(stack_h5_path.split('/')[-1]))
    except UnboundLocalError:
        stack_h5_path = None
        logging.info('  stack_h5_path: \n\t\t{} \n')
    
    try:
        logging.info('  soma_noise_h5_path:\n\t\t{}\n'.format(soma_noise_h5_path.split('/')[-1]))
    except UnboundLocalError:
        soma_noise_h5_path = None
        logging.info('  soma_noise_h5_path: \n\t\tNone \n')
    
    try:
        logging.info('  soma_chirp_h5_path: \n\t\t{}\n'.format(soma_chirp_h5_path.split('/')[-1]))
    except UnboundLocalError:
        soma_chirp_h5_path = None
        logging.info('  soma_chirp_h5_path: \n\t\tNone \n')    
    
    try:
        logging.info('  soma_lchirp_h5_path: \n\t\t{}\n'.format(soma_lchirp_h5_path.split('/')[-1]))
    except UnboundLocalError:
        soma_lchirp_h5_path = None
        logging.info('  soma_lchirp_h5_path: \n\t\tNone \n')
    

    logging.info('  noise_h5:')
    if len(noise_h5_paths) == 0:
        logging.info('  \tNo Noise files')
    else:
        for idx, noisefile in enumerate(noise_h5_paths):
            logging.info('  \t{}: {}'.format(idx, noisefile.split('/')[-1]))
        
    logging.info('  chirp_h5:')
    if len(chirp_h5_paths) == 0:
        logging.info('  \tNo Chirp files')
    else:
        for idx, chirpfile in enumerate(chirp_h5_paths):
            logging.info('  \t{}: {}'.format(idx, chirpfile.split('/')[-1]))
        
    logging.info('  lchirp_h5:')
    if len(lchirp_h5_paths) == 0:
        logging.info('  \tNo Local Chirp files')
    else:
        for idx, lchirpfile in enumerate(lchirp_h5_paths):
            logging.info('  \t{}: {}'.format(idx, lchirpfile.split('/')[-1])) 

    try:
        logging.info('  swc_path: \n\t\t{} \n'.format(swc_path.split('/')[-1]))
    except UnboundLocalError:
        stack_h5_path = None
        logging.info('  stack_h5_path: \n\t\t{} \n')

    try:
        logging.info('  headerfile_path: \n\t\t{} \n'.format(headerfile_path.split('/')[-1]))
    except UnboundLocalError:
        stack_h5_path = None
        logging.info('  headerfile_path: \n\t\t{} \n')

    data_dict = {

        "imaging_data_dir": imaging_data_dir,

        "stack_h5_path":stack_h5_path,
        
        "soma_noise_h5_path":soma_noise_h5_path,
        "soma_chirp_h5_path":soma_chirp_h5_path,
        "soma_lchirp_h5_path":soma_lchirp_h5_path,       
        "noise_h5_paths": noise_h5_paths,
        "chirp_h5_paths": chirp_h5_paths,
        "lchirp_h5_paths": lchirp_h5_paths,

        "swc_path": swc_path,

        "headerfile": headerfile_path,
    }
    
    logging.info('  Finished reading data paths.\n')

    return data_dict

def get_ttx_paths(rootdir, experimenter, expdate, expnum):

    exproot = rootdir + '/' + experimenter + '/' + expdate + '/' + str(expnum) + '/'
    imaging_data_dir = exproot + 'Pre/'
    morph_data_dir = exproot + 'Raw/'

    noise_h5_paths = []
    chirp_h5_paths = []
    lchirp_h5_paths = []    

    for file in os.listdir(exproot):
        if ('.ini' in file.lower()):
            headerfile_path = exproot + file

    for file in os.listdir(morph_data_dir):
        if ('.swc' in file.lower()):
            swc_path = morph_data_dir + file
    
    for file in os.listdir(imaging_data_dir):
#         logging.info(file)
        if ('stack.h5' in file.lower()):
            stack_h5_path = imaging_data_dir + file 
        
        if ('_s_dnoise' in file.lower()):
            soma_noise_h5_path = imaging_data_dir + file

        if ('_s_chirp' in file.lower()):
            soma_chirp_h5_path = imaging_data_dir + file

        if ('_s_lchirp' in file.lower()):
            soma_lchirp_h5_path = imaging_data_dir + file

        if ('dnoise' in file.lower() and '_s' not in file.lower() and 'ttx' in file.lower()):
            noise_h5_paths.append(imaging_data_dir + file)

        if ('_chirp' in file.lower() and '_s' not in file.lower() and 'ttx' in file.lower()):
            chirp_h5_paths.append(imaging_data_dir + file)
        
        if ('_lchirp' in file.lower() and '_s' not in file.lower() and 'ttx' in file.lower()):
            lchirp_h5_paths.append(imaging_data_dir + file)

    noise_h5_paths.sort()
    chirp_h5_paths.sort()
    lchirp_h5_paths.sort()

    logging.info('  Root Dir: \n\t\t{}'.format(rootdir))
    logging.info('  Imaging Data Dir: \n\t\t{}\n'.format(imaging_data_dir))
    logging.info('  Morph Data Dir: \n\t\t{}\n'.format(morph_data_dir))
    
    try:
        logging.info('  stack_h5_path: \n\t\t{} \n'.format(stack_h5_path.split('/')[-1]))
    except UnboundLocalError:
        stack_h5_path = None
        logging.info('  stack_h5_path: \n\t\t{} \n')
    
    try:
        logging.info('  soma_noise_h5_path:\n\t\t{}\n'.format(soma_noise_h5_path.split('/')[-1]))
    except UnboundLocalError:
        soma_noise_h5_path = None
        logging.info('  soma_noise_h5_path: \n\t\tNone \n')
    
    try:
        logging.info('  soma_chirp_h5_path: \n\t\t{}\n'.format(soma_chirp_h5_path.split('/')[-1]))
    except UnboundLocalError:
        soma_chirp_h5_path = None
        logging.info('  soma_chirp_h5_path: \n\t\tNone \n')    
    
    try:
        logging.info('  soma_lchirp_h5_path: \n\t\t{}\n'.format(soma_lchirp_h5_path.split('/')[-1]))
    except UnboundLocalError:
        soma_lchirp_h5_path = None
        logging.info('  soma_lchirp_h5_path: \n\t\tNone \n')
 
    logging.info('  noise_h5:')
    if len(noise_h5_paths) == 0:
        logging.info('  \tNo Noise files')
    else:
        for idx, noisefile in enumerate(noise_h5_paths):
            logging.info('  \t{}: {}'.format(idx, noisefile.split('/')[-1]))
        
    try:
        logging.info('  swc_path: \n\t\t{} \n'.format(swc_path.split('/')[-1]))
    except UnboundLocalError:
        stack_h5_path = None
        logging.info('  stack_h5_path: \n\t\t{} \n')

    try:
        logging.info('  headerfile_path: \n\t\t{} \n'.format(headerfile_path.split('/')[-1]))
    except UnboundLocalError:
        stack_h5_path = None
        logging.info('  headerfile_path: \n\t\t{} \n')

    data_dict = {

        "imaging_data_dir": imaging_data_dir,

        "stack_h5_path":stack_h5_path,
        
        "soma_noise_h5_path":soma_noise_h5_path,
        "soma_chirp_h5_path":soma_chirp_h5_path,
        "soma_lchirp_h5_path":soma_lchirp_h5_path,    
        "noise_h5_paths": noise_h5_paths,
        "chirp_h5_paths": chirp_h5_paths,
        "lchirp_h5_paths": lchirp_h5_paths,
        "swc_path": swc_path,

        "headerfile": headerfile_path,
    }

    
    logging.info('  Finished reading data paths.\n')

    return data_dict

def load_h5_data(file_name):
    """
    Helper function to load h5 file.
    """
    with h5py.File(file_name,'r') as f:
        return {key:f[key][:] for key in list(f.keys())}

# def load_stimulus(file_name):
#     with h5py.File(file_name)

def get_pixel_size_stack(stack):
    
    """
    Return the real length (in um) of each pixel point.
    """
    len_stack_x_pixel = stack['wParamsNum'][18]
    len_stack_x_um = 71.5 / stack['wParamsNum'][30]
    
    stack_pixel_size_x = len_stack_x_um / len_stack_x_pixel # stack_pixel_size_y == stack_pixel_size_x
    stack_pixel_size_z = stack['wParamsNum'][29]

    return np.array([stack_pixel_size_x, stack_pixel_size_x, stack_pixel_size_z])

def get_pixel_size_rec(rec):
    
    """
    Return the real length (in um) of each pixel point.
    """
    len_rec_x_pixel = rec['wDataCh0'].shape[0]
    len_rec_x_um = 71.5 / rec['wParamsNum'][30]
    
    rec_pixel_size = len_rec_x_um / len_rec_x_pixel
    
    return rec_pixel_size

def get_linestack(df_paths, stack_shape):
    coords = np.vstack(df_paths.path_stack)
    
    linestack = np.zeros(stack_shape)
    for c in coords:
        linestack[tuple(c)] = 1
        
    return linestack

def get_scale_factor(rec, stack):
    
    """
    get the scale factor from rec to stack, 
    e.g. scipy.misc.im
    (rec, size=scale_factor, interp='nearest')
    would make the rec into the same scale as stack. 
    """
    
    rec_pixel_size = get_pixel_size_rec(rec)
    stack_pixel_sizes = get_pixel_size_stack(stack)
    
    return rec_pixel_size / stack_pixel_sizes[0]

# def resize_roi(rec, stack):
#     return scipy.misc.imresize(rec['ROIs'], size=get_scale_factor(rec, stack), interp='nearest')

# def resize_rec(rec, stack):
    
#     reci = rec_preprop(rec)
    
#     return scipy.misc.imresize(reci, size=get_scale_factor(rec, stack), interp='nearest')


def resize_roi(rec, stack):
    
    output_shape = np.ceil(np.asarray(rec['ROIs'].shape) * get_scale_factor(rec, stack)).astype(int)
    
    return np.round(resize(rec['ROIs'], output_shape=output_shape, order=0, mode='constant'))
#     return resize(rec['ROIs'], output_shape=output_shape, order=0, mode='constant')

def resize_rec(rec, stack):
    
    reci = rec_preprop(rec)
    
    output_shape = np.ceil(np.asarray(reci.shape) * get_scale_factor(rec, stack)).astype(int)

    return resize(reci, output_shape=output_shape, order=0, mode='constant')

def rotate_rec(rec, stack,angle_adjust=0):
    
    ang_deg = rec['wParamsNum'][31] + angle_adjust# ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratoate angle (radian)
    
    rec_rec = resize_rec(rec, stack)
    rec_rot = rotate(rec_rec, ang_deg, resize=True, order=1, cval=rec_rec.min())
    
    (shift_x, shift_y) = 0.5 * (np.array(rec_rot.shape) - np.array(rec_rec.shape))
    (cx, cy) = 0.5 * np.array(rec_rec.shape)
    
    px, py = (0, 0) # origin
    px -= cx
    py -= cy
    
    xn = px * np.cos(ang_rad) - py*np.sin(ang_rad)
    yn = px * np.sin(ang_rad) + py*np.cos(ang_rad)
    
    xn += (cx + shift_x)
    yn += (cy + shift_y)    
    
    # the shifted origin after rotation
    
    return rec_rot, (xn, yn)

def rec_preprop(rec):
    
    reci = rec['wDataCh0'].mean(2)
    reci[:4, :] = reci.mean() - 0.5*reci.std()
    
    return reci

def rotate_roi(rec, stack, angle_adjust=0):
    
    ang_deg = rec['wParamsNum'][31] + angle_adjust # ratoate angle (degree)
    ang_rad = ang_deg * np.pi / 180 # ratoate angle (radian)
    
    rec_rois = resize_roi(rec, stack)
    rec_rois_rot = rotate(rec_rois, ang_deg, cval=1, order=0, resize=True)
    rec_rois_rot = np.ma.masked_where(rec_rois_rot == 1, rec_rois_rot)

    (shift_x, shift_y) = 0.5 * (np.array(rec_rois_rot.shape) - np.array(rec_rois.shape))
    (cx, cy) = 0.5 * np.array(rec_rois.shape)
    
    labels = np.unique(rec_rois)[:-1][::-1]
    # reverse the lables to keep consistent with the labels of raw traces

    px = [np.vstack(np.where(rec_rois == i)).T[:, 0].mean() for i in labels] 
    py = [np.vstack(np.where(rec_rois == i)).T[:, 1].mean() for i in labels]

    px -= cx
    py -= cy
    
    xn = px * np.cos(ang_rad) - py*np.sin(ang_rad)
    yn = px * np.sin(ang_rad) + py*np.cos(ang_rad)
    
    xn += (cx + shift_x)
    yn += (cy + shift_y)
    
    return rec_rois_rot, np.vstack([xn, yn]).T

def rel_position_um(soma, d):
    
    """
    Relative position between dendrites and soma in um.
    
    Return
    ======
    
    array([YCoord_um, XCoord_um, ZCoord_um])
    """
    
    return soma['wParamsNum'][26:29] - d['wParamsNum'][26:29]

def roi_matching(image, template):
    
    result = match_template(image, template)
    ij = np.unravel_index(np.argmax(result), result.shape)
    
    return np.array(ij)

def calibrate_one_roi(coords_xy, linestack):
    
    x_o = coords_xy[0] # roi_x_original
    y_o = coords_xy[1] # roi_y_original

    search_scope = 0
    offset = np.where(linestack[x_o:x_o+1, y_o:y_o+1] == 1)

    while offset[2].size == 0:
        search_scope +=1
        offset = np.where(linestack[x_o-search_scope:x_o+search_scope+1, y_o-search_scope:y_o+search_scope+1] == 1)

    z_o = np.mean(offset[2]).astype(int)  # roi_z_original, this is a guess

    x_c = np.arange(x_o-search_scope,x_o+search_scope+1)[offset[0]]
    y_c = np.arange(y_o-search_scope,y_o+search_scope+1)[offset[1]]
    z_c = offset[2]

    candidates = np.array([np.array([x_c[i], y_c[i], z_c[i]]) for i in range(len(x_c))])
    origins = np.array([x_o, y_o, z_o])

    x, y, z = candidates[np.argmin(np.sum((candidates - origins) ** 2, 1))]

    return np.array([x,y,z])

def on_which_path(df_paths, point):
    
    result_path = df_paths[df_paths.path_stack.apply(lambda x: (x == point).all(1).any())]
    path_id = result_path.index[0]
    
    return path_id

def get_loc_on_path_stack(df_paths, point):
    
    loc = [i for i in df_paths.path_stack.apply(lambda x: np.where((x == point).all(1))[0]) if len(i) != 0][0]
    
    return loc[0]

def get_segment_length(arr):
    
    return np.sum(np.sqrt(np.sum((arr[1:] - arr[:-1])**2, 1)))

def get_dendritic_distance_to_soma(df_paths, path_id, loc_on_path):
    
    length_all_paths = sum(df_paths.loc[df_paths.loc[path_id]['back_to_soma']]['real_length'])
    length_to_reduce = get_segment_length(df_paths.loc[path_id].path[loc_on_path:])
    
    return length_all_paths-length_to_reduce

def get_euclidean_distance_to_one_point(roi_pos, point_pos):
    
    return np.sqrt(np.sum((roi_pos -point_pos) ** 2))


def get_density_center(df_paths, soma, Z):    
    
    from scipy.ndimage.measurements import center_of_mass
    def change_coordinate(coords, origin_size, new_size):

        coords_new = np.array(coords) * (max(new_size) - min(new_size)) / max(origin_size) - max(new_size) 

        return coords_new

    xy = np.vstack(df_paths.path)[:, :2] - soma[:2]
    lim_max = int(np.ceil((xy.T).max() / 20) * 20)
    lim_min = int(np.floor((xy.T).min() / 20) * 20)
    lim = max(abs(lim_max), abs(lim_min))

    density_center = center_of_mass(Z)[::-1]
    density_center = change_coordinate(density_center, (0, max(Z.shape)), (-lim, lim))
    density_center += soma[:2]
    
    return density_center


# helpers for pariwise

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_btw_node(roi_0_pos, roi_1_pos, node):
    
    v0 = unit_vector(roi_0_pos - node)
    v1 = unit_vector(roi_1_pos - node)
    
    return np.degrees(np.arccos(np.clip(np.dot(v0, v1), -1.0, 1.0)))
    

# def get_cntr_interception(cntr0, cntr1):
    
#     if cntr0.shape[0] > cntr1.shape[0]:
#         sCntr = cntr1.copy()
#         bCntr = cntr0.copy()
#     else:
#         sCntr = cntr0.copy()
#         bCntr = cntr1.copy()
    
#     sCntrPoly = Polygon(sCntr)
#     bCntrPoly = Polygon(bCntr)
    
#     sCntr_area = sCntrPoly.area
#     bCntr_area = bCntrPoly.area
    
#     check_intercept =  sCntrPoly.intersects(bCntrPoly)
    
#     if check_intercept:
        
#         CntrInpt = sCntrPoly.intersection(bCntrPoly)
        
#         if CntrInpt.type == 'Polygon':
#             inner_cntr_list = [np.asarray(CntrInpt.boundary.coords)]
#             overlap_area = CntrInpt.area
#         elif CntrInpt.type == 'MultiPolygon':
#             inner_cntr_list = []
#             overlap_area = 0
#             for i in np.arange(len(CntrInpt.geoms)):
#                 inner_cntr_list.append(np.asarray(CntrInpt.geoms[i].boundary.coords))
#                 overlap_area += CntrInpt.geoms[i].area
#     else:
#         inner_cntr_list = [np.nan]
#         overlap_area = 0
        
#     return inner_cntr_list, sCntr_area/1000, bCntr_area/1000, overlap_area/1000, overlap_area/sCntr_area

def get_cntr_interception(roi_0_cntr, roi_1_cntr):
    
    inner_cntr_list_all = []
    overlap_index_all = []
    for cntr0 in roi_0_cntr:
        for cntr1 in roi_1_cntr:
            if cntr0.shape[0] > cntr1.shape[0]:
                sCntr = cntr1.copy()
                bCntr = cntr0.copy()
            else:
                sCntr = cntr0.copy()
                bCntr = cntr1.copy()

            sCntrPoly = Polygon(sCntr)
            bCntrPoly = Polygon(bCntr)

            sCntr_area = sCntrPoly.area
            bCntr_area = bCntrPoly.area

            check_intercept =  sCntrPoly.intersects(bCntrPoly)

            if check_intercept:

                CntrInpt = sCntrPoly.intersection(bCntrPoly)

                if CntrInpt.type == 'Polygon':
                    inner_cntr_list = [np.asarray(CntrInpt.boundary.coords)]
                    overlap_area = CntrInpt.area
                elif CntrInpt.type == 'MultiPolygon':
                    inner_cntr_list = []
                    overlap_area = 0
                    for i in np.arange(len(CntrInpt.geoms)):
                        inner_cntr_list.append(np.asarray(CntrInpt.geoms[i].boundary.coords))
                        overlap_area += CntrInpt.geoms[i].area
                        
            else:
                inner_cntr_list = [np.nan]
                overlap_area = 0
                
            overlap_index_all.append(overlap_area/sCntr_area)
            inner_cntr_list_all.append(inner_cntr_list)
            
    return inner_cntr_list_all, np.mean(overlap_index_all)
