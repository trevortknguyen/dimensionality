from .som import SelfOrganizingMap2D
from .som import SelfOrganizingMap3D
from .autoencoder import Autoencoder3D

import numpy as np

from neural_dimensionality_reduction.plots import get_color_3d

def get_summary_statistics(spikes):
    mu = np.mean(np.mean(spikes, axis=0))
    sigma = np.mean(np.sqrt(np.mean(np.square(spikes - mu), axis=0)))
    return mu, sigma

   
def get_transformed_colors_3d(model, spikes):
    transformed = np.empty((spikes.shape[0], 3))
    colors = np.empty(spikes.shape[0], dtype='object')
    
    for i in range(spikes.shape[0]):
        x, y, z = model.transform1(spikes[i])
        transformed[i] = x, y, z
        colors[i] = get_color_3d(int(x/model.shape[0]*256), int(y/model.shape[1]*256), int(z/model.shape[2]*256))
        
    return transformed, colors
    

