import cv2
import logging
import astropy.modeling
import numpy as np
import scipy as sp
import pandas as pd
import scipy.ndimage as ndimage

from sklearn.utils.extmath import randomized_svd
from matplotlib import _cntr as cntr

 

def znorm(data):
    return (data - data.mean())/data.std()

def interpolate_weights(tracetime, traces_znorm, triggers):

    data_interp = sp.interpolate.interp1d(
        tracetime.flatten(), 
        traces_znorm.flatten(),
        kind = 'linear'
    ) (triggers)
    
    return znorm(data_interp)

def lag_weights(weights,nLag):
    lagW = np.zeros([weights.shape[0],nLag])
    
    for iLag in range(nLag):
        lagW[iLag:-nLag+iLag,iLag] = weights[nLag:]/nLag
        
    return lagW

def get_sta(*data, stimulus, source='trace_raw'):
    
    (rec_id, roi_id, rf, 
     tracetime, triggertime, 
     traces_raw, traces_znorm) = data

    if source == 'igor':
        return [rf]
    elif source == 'traces_raw':
        weights = interpolate_weights(tracetime, 
                                      znorm(traces_raw.flatten()), 
                                      triggertime)
    elif source == 'traces_znorm':
        weights = interpolate_weights(tracetime, 
                                      traces_znorm, 
                                      triggertime)
    else:
        logging.info('  Wrong source!')
        return None

    lagged_weights = lag_weights(weights, 5)
    stimulus = stimulus[:, :len(weights)]
    sta = stimulus.dot(lagged_weights)
    U,S,Vt = randomized_svd(sta, 3)
    
    return [U[:, 0].reshape(15,20)]
    

def smooth_rf(rf, sigma):
    return ndimage.gaussian_filter(rf, sigma=(sigma, sigma), order=0)

# def upsample_rf(rf, rf_pixel_size, stack_pixel_size):
#     scale_factor = rf_pixel_size/stack_pixel_size
#     return sp.misc.imresize(rf, size=scale_factor, interp='bilinear', mode='F')

def upsample_rf(rf, rf_pixel_size):
    return sp.misc.imresize(rf, size=rf_pixel_size, interp='bilinear', mode='F')

def rescale_data(data):
    return (data - data.min()) / (data.max() - data.min())

def standardize_data(data):
    return (data-data.mean()) / data.std()

def get_contour(data, threshold=3):
    
    data = standardize_data(data)

    x, y = np.mgrid[:data.shape[0], :data.shape[1]]
    c = cntr.Cntr(x,y, data)
    
    SD = data.std()
    res = c.trace(data.max() - threshold * SD)

    if len(res) < 2:
        rf_cntr, rf_size = np.nan, np.nan
    else:
        rf_size_list = []
        for i in range(int(len(res)/ 2)):
            rf_cntr = res[i]

            rf_size = cv2.contourArea(rf_cntr.astype(np.float32))
            rf_size_list.append(rf_size)
            
        largest = np.argmax(rf_size_list)
        rf_cntr = res[largest]
        rf_size = rf_size_list[largest]
        
    return pd.Series([rf_cntr, rf_size])

def gaussian_fit(rf):
    
    x = np.arange(0, rf.shape[0])
    y = np.arange(0, rf.shape[1])
    Y, X = np.meshgrid(x, y)

    g_init = astropy.modeling.models.Gaussian2D(amplitude=rf.max(), x_mean=rf.shape[1]/2, y_mean=rf.shape[0]/2)
    fitter = astropy.modeling.fitting.SLSQPLSQFitter()
    g = fitter(g_init, X, Y, rf.T, verblevel=0)

    return g(X, Y).T