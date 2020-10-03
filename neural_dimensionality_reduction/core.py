import os
import netCDF4 as nc
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

data_dir = os.environ['DATA_DIR']

def get_formatted_data(data_dir, animal_id = 'remy_35_02', tetrode_id = 1):
    marks_fn = f'{data_dir}/{animal_id}_marks.nc'
    position_fn = f'{data_dir}/{animal_id}_position_info.csv'

    ds = nc.Dataset(marks_fn)
    neural = ds['__xarray_dataarray_variable__']

    na = np.array(neural[:,:,tetrode_id]) # (timestamps, channels, tetrodes)

    n_timestamps = na.shape[0]
    
    spikes = na.reshape((n_timestamps, 4, 1))
    
    df = pd.read_csv(position_fn)
    states = df['linear_position'].values.reshape((n_timestamps,))
    
    assert spikes.shape[0] == states.shape[0]
    return spikes.astype('float32'), states.astype('float32')

def get_spikes_data(data_dir, animal_id = 'remy_35_02', tetrode_id = 1):
    marks_fn = f'{data_dir}/{animal_id}_marks.nc'
    position_fn = f'{data_dir}/{animal_id}_position_info.csv'

    ds = nc.Dataset(marks_fn)
    neural = ds['__xarray_dataarray_variable__']

    na = np.array(neural[:,:,tetrode_id]) # (timestamps, channels, tetrodes)
    spikes_bool = np.logical_not(np.isnan(na)).any(axis=1)
    
    spikes_idx = np.argwhere(spikes_bool)

    n_spikes = spikes_idx.shape[0] # important
    
    spikes_unshaped = na[spikes_idx]
    
    # spikes
    spikes = spikes_unshaped.reshape((n_spikes, 4))
    
    df = pd.read_csv(position_fn)
    states = df['linear_position'][spikes_idx.reshape(n_spikes)].values.reshape((n_spikes,1))
    
    assert spikes.shape[0] == states.shape[0]
    return spikes.astype('int16'), states.astype('float32')


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
    
def get_marks_data(data_dir, animal_id = 'remy_35_02', tetrode_id = 1):
    spikes, states = get_spikes_data(data_dir, animal_id, tetrode_id)
    marks = transform_spikes_to_marks(spikes)
    assert marks.shape[0] == states.shape[0]
    return marks, states

def calculate_mse_causal(results, position, time_ind):
    max_values = np.argmax(np.nan_to_num(results.causal_posterior.values), axis=1)
    return np.mean(np.square(max_values-position[time_ind]))
    