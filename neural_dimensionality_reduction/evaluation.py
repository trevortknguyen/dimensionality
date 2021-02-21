import numpy as np

from replay_trajectory_classification import ClusterlessDecoder
from replay_trajectory_classification.state_transition import estimate_movement_var
from replay_trajectory_classification.misc import NumbaKDE

def get_decoder():
    model_kwargs = {
    'bandwidth': np.array([1.0, 1.0, 1.0, 1.0, 12.5]) # amplitude 1, amplitude 2, amplitude 3, amplitude 4, position
    }

    decoder = ClusterlessDecoder(model_kwargs=model_kwargs)
    
    return decoder

def calculate_mse_causal(results, position, time_ind):
    max_values = np.argmax(np.nan_to_num(results.causal_posterior.values), axis=1)
    return np.mean(np.square(max_values-position[time_ind]))

def compare_distributions():
    pass
