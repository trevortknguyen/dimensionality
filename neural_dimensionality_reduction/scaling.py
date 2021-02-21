import os
import netCDF4 as nc
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

def scale_identity(spikes):
    return spikes

def scale_standard_scaler(spikes):
    return StandardScaler().fit_transform(spikes)

def scale_custom(spikes):
    '''
    Scales the spikes by dividing 
    '''
    tetrode_means = np.mean(spikes, axis=1)

    broadcasted_means = np.empty(spikes.shape)
    broadcasted_means[:,0] = tetrode_means
    broadcasted_means[:,1] = tetrode_means
    broadcasted_means[:,2] = tetrode_means
    broadcasted_means[:,3] = tetrode_means
    
    return spikes / broadcasted_means

def pca_transform(scaled_spikes):
    pca = PCA(n_components=3)
    return pca.fit_transform(scaled_spikes)

def transform_spikes_to_marks(spikes):
    scaled_spikes = scale_custom(spikes)
    marks = pca_transform(scaled_spikes)

    assert marks.shape[0] == spikes.shape[0]
    return marks
    
