import cv2
import logging
import astropy.modeling
import numpy as np
import scipy as sp
import pandas as pd
import scipy.ndimage as ndimage

from sklearn.utils.extmath import randomized_svd
# from matplotlib import _cntr as cntr
import matplotlib.pyplot as plt

# from rfest import ASD
from rfest import splineLG, build_design_matrix, get_spatial_and_temporal_filters
from rfest._utils import softthreshold

def znorm(data):
    return (data - data.mean())/data.std()

def interpolate_weights(tracetime, traces_znorm, triggers):

    data_interp = sp.interpolate.interp1d(
        tracetime.flatten(), 
        traces_znorm.flatten(),
        kind = 'linear', fill_value='extrapolate',
    ) (triggers)
    
    return znorm(data_interp)

def lag_weights(weights,nLag):
    lagW = np.zeros([weights.shape[0],nLag])
    
    for iLag in range(nLag):
        lagW[iLag:-nLag+iLag,iLag] = weights[nLag:]/nLag
        
    return lagW

def upsampled_stimulus(trace, tracetime, triggertime, stim):
    
    valid_duration = np.logical_and(tracetime > triggertime[0], tracetime < triggertime[-1])

    trace_valid = trace[valid_duration]
    tracetime_valid = tracetime[valid_duration]

    y = znorm(trace_valid.copy())
    y = np.gradient(y)
    
    frames = np.vstack([triggertime[:-1], triggertime[1:]]).T

    num_repeats = []
    for i in range(len(frames)):
        num_repeats.append(sum((tracetime > frames[i][0]).astype(int) * (tracetime <= frames[i][1]).astype(int)))
    num_repeats = np.hstack(num_repeats)

    X = np.repeat(stim[:len(frames)], num_repeats, axis=0)
    
    return X, y


def get_rf(*data, stimulus, df, lambd):
    
    (rec_id, roi_id, 
     tracetime, triggertime, 
     traces_raw) = data

    # dims = [5, 20, 15]
    # y = interpolate_weights(tracetime, 
    #                              znorm(traces_raw.flatten()), 
    #                              triggertime)


    # y = y[:1500]
    # y = np.gradient(y) # take the derivative of the calcium trace
    # stimulus = stimulus[:len(y), :]
    # X = build_design_matrix(stimulus, dims[0])
    
    dims = [20, 20, 15]
    X, y = upsampled_stimulus(traces_raw, tracetime, triggertime, stimulus)
    X = build_design_matrix(X, dims[0])

    spl = splineLG(X, y, dims, df=df, smooth='cc', compute_mle=False)
    w = softthreshold(spl.w_spl, lambd)

    sRF, tRF = get_spatial_and_temporal_filters(w, dims)

    if np.argmax([np.abs(sRF.max()), np.abs(sRF.min())]):
        sRF = -sRF
        tRF = -tRF

    return [sRF.T, tRF, w]    
        
def smooth_rf(rf, sigma):
    return ndimage.gaussian_filter(rf, sigma=(sigma, sigma), order=0)

def upsample_rf(rf, rf_pixel_size, stack_pixel_size):
    scale_factor = rf_pixel_size/stack_pixel_size
    return sp.misc.imresize(rf, size=scale_factor, interp='bilinear', mode='F')

def rescale_data(data):
    return (data - data.min()) / (data.max() - data.min())

def standardize_data(data):
    return (data-data.mean()) / data.std()

def get_contour(data, levels, stack_pixel_size, 
                rf_pixel_size=30):

    data = rescale_data(data)
    CS = plt.contour(data, levels=levels)
    plt.clf()
    plt.close()

    res = 0
    for i in range(len(levels)):

        ps = CS.collections[i].get_paths()
        
        all_cntrs = [p.vertices for p in ps]

        cntrs_size = [cv2.contourArea(cntr.astype(np.float32))*stack_pixel_size**2/1000 for cntr in all_cntrs]

        good_cntrs = [cntr[:, ::-1] for cntr in all_cntrs if (cntr[0] == cntr[-1]).all() and cv2.contourArea(cntr.astype(np.float32))*stack_pixel_size**2/1000 > 1]
        good_cntrs_size = [cv2.contourArea(cntr.astype(np.float32))*stack_pixel_size**2/1000 for cntr in good_cntrs]

        if i == 0:
            res = pd.Series([good_cntrs, good_cntrs_size])
        else:
            tmp = pd.Series([good_cntrs, good_cntrs_size])
            res = res.append(tmp)

    res.index = np.arange(len(levels) * 2)
    return res

def gaussian_fit(rf):
    
    x = np.arange(0, rf.shape[0])
    y = np.arange(0, rf.shape[1])
    Y, X = np.meshgrid(x, y)

    g_init = astropy.modeling.models.Gaussian2D(amplitude=rf.max(), x_mean=rf.shape[1]/2, y_mean=rf.shape[0]/2)
    fitter = astropy.modeling.fitting.SLSQPLSQFitter()
    g = fitter(g_init, X, Y, rf.T, verblevel=0)

    return g(X, Y).T

def get_irregular_index(cnts):
    irregular_index = []
    
    if len(cnts) == 0: return 1
    
    for j, cnt in enumerate(cnts):
        hull = cv2.convexHull(cnt.astype(np.float32)).flatten().reshape(-1, 2)
        hull = np.vstack([hull, hull[0]])
        
        RFarea = cv2.contourArea(cnt.astype(np.float32))
        CHarea = cv2.contourArea(hull.astype(np.float32))

        irregular_index.append((CHarea - RFarea) / CHarea)
            
    return np.max(irregular_index)
