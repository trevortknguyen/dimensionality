import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.express as px

def get_color_2d(x, y):
    assert x >= 0
    assert x <= 255
    assert y >= 0
    assert y <= 255
    
    xhex = hex(x)[2:].zfill(2)
    yhex = hex(y)[2:].zfill(2)
    return f'#ff{xhex}{yhex}'.upper()

def get_color_3d(x, y, z):
    assert x >= 0
    assert x <= 255
    assert y >= 0
    assert y <= 255
    assert z >= 0
    assert z <= 255
    
    xhex = hex(x)[2:].zfill(2)
    yhex = hex(y)[2:].zfill(2)
    zhex = hex(z)[2:].zfill(2)
    return f'#{zhex}{xhex}{yhex}'.upper()

def plot_colormap_2d(model):
    colors = [ get_color_2d(int(x/model.shape[0]*256), int(y/model.shape[1]*256)) for (x, y) in list(zip(model.xs, model.ys))]
    
    return plt.scatter(x=model.xs,
                       y=model.ys,
                       color=colors)

def plot_colormap_3d(model):
    colors = [
        get_color_3d(int(x/model.shape[0]*256), int(y/model.shape[1]*256), int(z/model.shape[2]*256))
        for (x, y, z) in list(zip(model.xs, model.ys, model.zs))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model.xs, model.ys, model.zs, c=colors)
    return fig, ax

def plot_heatmap_2d(model, transformed):
    return px.density_heatmap(x=transformed[:,0], y = transformed[:,1], nbinsx=model.shape[0], nbinsy=model.shape[1])

def get_counts_3d(model, transformed):
    counts = np.zeros((model.shape[0], model.shape[1], model.shape[2]))
    for i in range(transformed.shape[0]):
        x, y, z = transformed[i].astype('int')
        counts[x,y,z] += 1
    return counts

def plot_heatmap_3d(model, transformed):
    counts = get_counts_3d(model, transformed).reshape(-1)
    
    return px.scatter_3d(x = model.xs, y = model.ys, z = model.zs, color=counts,
              range_color=[np.percentile(counts, 5), np.percentile(counts, 99)])

def plot_tetrode_colored(spikes, colors):
    '''
    Only works on 4-d original data.
    '''

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True, sharex=True, sharey=True)

    axes[0, 0].scatter(x=spikes[:,0],
               y = spikes[:,1],
               color=colors)
    axes[0, 0].set_xlabel('electrode 0')
    axes[0, 0].set_ylabel('electrode 1')

    axes[0, 1].scatter(x=spikes[:,2],
               y = spikes[:,1],
               color=colors)
    axes[0, 1].set_xlabel('electrode 2')
    axes[0, 1].set_ylabel('electrode 1')

    axes[1, 0].scatter(x=spikes[:,0],
               y = spikes[:,3],
               color=colors)
    axes[1, 0].set_xlabel('electrode 0')
    axes[1, 0].set_ylabel('electrode 3')
    
    return fig, axes