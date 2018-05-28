import cv2
import logging
import astropy.modeling
import numpy as np
import scipy as sp
import pandas as pd
import scipy.ndimage as ndimage

from sklearn.utils.extmath import randomized_svd
from matplotlib import _cntr as cntr
import matplotlib.pyplot as plt

from rfest import ASD

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

def get_sta(*data, stimulus, methods, num_iters):
    
    (rec_id, roi_id, 
     tracetime, triggertime, 
     traces_raw) = data

    weights = interpolate_weights(tracetime, 
                                  znorm(traces_raw.flatten()), 
                                  triggertime)


    weights = weights[:1500]
    weights = np.gradient(weights) # take the derivative of the calcium trace
    stimulus = stimulus[:len(weights), :]

    if methods=='mle':
        lagged_weights = lag_weights(weights, 5)
        sta = stimulus.T.dot(lagged_weights)
        U,S,Vt = randomized_svd(sta, 3)
        sRF_opt = U[:, 0].reshape(15,20)
        tRF_opt = Vt[0]
        
    elif methods=='asd':

        def callback(params, t, g):
            if (t+1) % num_iters == 0:
                print("Rec {} ROI{}: {}/{}.".format(rec_id, roi_id, t+1, num_iters))

        asd = ASD(stimulus, weights, rf_dims=(15,20,5))
        asd.fit(initial_params=([2.29,-0.80,2.3]),num_iters=num_iters,callback=callback)
        sta = asd.w_opt.reshape(15,20,5)
        sRF_opt = asd.sRF_opt
        tRF_opt = asd.tRF_opt
        
    return [sRF_opt, tRF_opt, sta]
    

def smooth_rf(rf, sigma):
    return ndimage.gaussian_filter(rf, sigma=(sigma, sigma), order=0)

def upsample_rf(rf, rf_pixel_size):
    return sp.misc.imresize(rf, size=rf_pixel_size, interp='bilinear', mode='F')

def rescale_data(data):
    return (data - data.min()) / (data.max() - data.min())

def standardize_data(data):
    return (data-data.mean()) / data.std()

def get_contour(data):

    data = rescale_data(data)

    CS = plt.contour(normalize(data), levels=np.linspace(0, 1, 21))
    ps = CS.collections[-8].get_paths()
    all_cntrs = [p.vertices for p in ps]
    good_cntrs = [cntr for cntr in all_cntrs if (cntr[0] == cntr[-1]).all() and cv2.contourArea(cntr.astype(np.float32)) > 2]
    good_cntrs_size = [cv2.contourArea(cntr.astype(np.float32)) for cntr in good_cntrs]

    return pd.Series([good_cntrs, good_cntrs_size])

def get_contour(data, threshold=2):
    
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

    if (rf_cntr[0] != rf_cntr[-1]).any():
        rf_cntr = np.vstack([rf_cntr, rf_cntr[0]])
        
    return pd.Series([rf_cntr, rf_size])

def gaussian_fit(rf):
    
    x = np.arange(0, rf.shape[0])
    y = np.arange(0, rf.shape[1])
    Y, X = np.meshgrid(x, y)

    g_init = astropy.modeling.models.Gaussian2D(amplitude=rf.max(), x_mean=rf.shape[1]/2, y_mean=rf.shape[0]/2)
    fitter = astropy.modeling.fitting.SLSQPLSQFitter()
    g = fitter(g_init, X, Y, rf.T, verblevel=0)

    return g(X, Y).T

def quality_index(x, y):
    
    def norm(x):
        return (x - x.min()) / (x.max() - x.min())
    
    x = norm(x.ravel())
    y = norm(y.ravel())
    
    n = len(x)
    
    mx = np.mean(x)
    my = np.mean(y)
    
    varx = np.var(x)
    vary = np.var(y)

    stdx = np.sqrt(varx)
    stdy = np.sqrt(vary)
    
    varxy = np.sum((x-mx) * (y-my)) / (n-1)
    
    loss_corr = varxy / (stdx * stdy)
    
    m_dtt = 2 * mx * my / (mx **2 + my ** 2)
    v_dtt = 2 * stdx * stdy / (varx + vary)
    
    q = loss_corr * m_dtt * v_dtt
    
    return q, (loss_corr, m_dtt, v_dtt)