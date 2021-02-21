import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import h5py

data_dir = os.environ['DATA_DIR']

def transform_formatted_data_multiple_tetrodes(multiunits, models):
    '''
    Assumes that the model takes it to 3.
    Assumes the model is 24.

    Stupid assumptions, I know.
    '''
    assert multiunits.shape[2] == len(models)

    transformed = np.empty((multiunits.shape[0], 3, multiunits.shape[2]))
    transformed[:] = np.nan;

    # we need to train on every single one
    for i in range(multiunits.shape[2]):
        # transform all the things
        spikes = multiunits[:,:,i].reshape((multiunits.shape[0], multiunits.shape[1], 1))

        print(f'spikes: {spikes.shape}')

        spikes_idx1 = np.isnan(spikes)
        print(spikes_idx1.shape)
        spikes_idx2 = np.logical_not(spikes_idx1)
        print(spikes_idx2.shape)
        spikes_idx3 = spikes_idx2.any(axis=1)
        print(spikes_idx3.shape)
        spikes_idx4 = np.argwhere(spikes_idx3[:,0])
        print(spikes_idx4.shape)
        spikes_idx5 = spikes_idx4[:,0]
        print(spikes_idx5.shape)

        spikes_idx = spikes_idx5

        # spikes_idx = np.argwhere(np.logical_not(np.isnan(spikes)).any(axis=1)[:,i])[:,i]

        for j in spikes_idx:
            x, y, z = models[i].transform1(spikes[j, :, 0])
            transformed[j, :, i] = x, y, z
        
    return transformed

def train_multiple_transformation_models_independently(multiunits, create_model):
    '''
    Assumes the model creation matches the dimensionality.

    model.train1(spike) should work.

    create_model should take no arguments and create a model
    '''
    models = []

    for i in range(24):
        model = create_model()
        
        spikes = extract_spikes_tetrode_from_multiunits(multiunits, i)
        assert len(spikes.shape) == 2
        assert spikes.shape[1] == 4
        for spike in spikes:
            model.train1(spike)
        
        models.append(model)
    return models

def extract_spikes_tetrode_from_multiunits(multiunits, tetrode_id):
    na = np.array(multiunits[:,:,tetrode_id])
    spikes_bool = np.logical_not(np.isnan(na)).any(axis=1) 
    spikes_idx = np.argwhere(spikes_bool) 
    n_spikes = spikes_idx.shape[0]
    spikes_unshaped = na[spikes_idx]
    spikes = spikes_unshaped.reshape((n_spikes, 4))

    # same shape spike
    assert spikes.shape[1] == multiunits.shape[1]

    return spikes

def get_formatted_data_full(data_dir, animal_id = 'remy_35_02'):
    '''
    Given a data directory and filename, this will load the files in the format
    for direct use in the `replay_trajectory_classification`. It uses the
    original spike features which are likely the max amplitudes of the tetrode.

    Not all spikes are happening at the same time.
    '''
    marks_fn = f'{data_dir}/{animal_id}_marks.nc'
    position_fn = f'{data_dir}/{animal_id}_position_info.csv'

    ds = nc.Dataset(marks_fn)
    neural = ds['__xarray_dataarray_variable__']

    spikes = np.array(neural) # (timestamps, channels, tetrodes)
    n_timestamps = spikes.shape[0]

    df = pd.read_csv(position_fn)
    states = df['linear_position'].values.reshape((n_timestamps,))
    
    assert spikes.shape[0] == states.shape[0]
    return spikes.astype('float32'), states.astype('float32')

def transform_formatted_data(model, spikes):
    '''
    Given a model, transforms data formatted for the clusterless decoder into
    the same format but a lower dimension.

    Input: (n_timestamps, dim, 1)
    Output: (n_timestamps, low_dom, 1)

    Assume the model is already trained.
    '''
    transformed = np.empty((spikes.shape[0], 3, 1))
    transformed[:] = np.nan;

    spikes_idx = np.argwhere(np.logical_not(np.isnan(spikes)).any(axis=1)[:,0])[:,0]

    for i in spikes_idx:
        x, y, z = model.transform1(spikes[i, :, 0])
        transformed[i, :, 0] = x, y, z
        
    return transformed

def get_formatted_data(data_dir, animal_id = 'remy_35_02', tetrode_id = 1):
    '''
    Given a data directory and filename, this will load the files in the format
    for direct use in the `replay_trajectory_classification`. It uses the
    original spike features which are likely the max amplitudes of the tetrode.

    But, this only does one tetrode at a time.
    '''
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
    '''
    What format is this?
    '''
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

def get_marks_data(data_dir, animal_id = 'remy_35_02', tetrode_id = 1):
    '''
    What format is this?
    '''
    spikes, states = get_spikes_data(data_dir, animal_id, tetrode_id)
    marks = transform_spikes_to_marks(spikes)
    assert marks.shape[0] == states.shape[0]
    return marks, states

def insert_hdf5_file(filename, tetrode_id, scheme_name, transformed):
    '''
    Used to store the results to disk to make things easier.
    Assumes floats and ndarray
    '''
    with h5py.File(filename, 'r+') as f:
        ds_transformed = f.create_dataset(f'tetrode_{tetrode_id}/marks_{scheme_name}', transformed.shape, dtype='f')
        ds_transformed[:] = transformed

def retrieve_hdf5_file(filename, tetrode_id, scheme_name):
    with h5py.File(filename, 'r') as f:
        ds = f[f'tetrode_{tetrode_id}/marks_{scheme_name}']
        data = ds[()]
    return data

def retrieve_hdf5_file_positions(filename, tetrode_id):
    with h5py.File(filename, 'r') as f:
        ds = f[f'tetrode_{tetrode_id}/positions']
        data = ds[()]
    return data
