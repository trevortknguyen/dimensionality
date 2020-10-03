import numpy as np

from neural_dimensionality_reduction.plots import get_color_3d

def get_summary_statistics(spikes):
    mu = np.mean(np.mean(spikes, axis=0))
    sigma = np.mean(np.sqrt(np.mean(np.square(spikes - mu), axis=0)))
    return mu, sigma

class SelfOrganizingMap2D:
    '''
    This is a 2D map.
    '''
    def __init__(self, shape, mu, sigma, weights=None):
        self.shape = shape

        if weights is None:
            self.weights = np.random.normal(mu, sigma, shape)
        else:
            assert self.shape == weights.shape
            self.weights = weights


        self.xs = np.repeat(np.arange(0, shape[0]), shape[1])
        self.ys = np.tile(np.arange(0, shape[1]), shape[0])
    
    def distance_function(x0, y0, x, y):
        dx, dy = x0 - x, y0 - y
        return dx*dx + dy*dy

    def sigma_function(t):
        return 1

    def change_function(x0, y0, x, y):
        d = SelfOrganizingMap2D.distance_function(x0, y0, x, y)
        s = SelfOrganizingMap2D.sigma_function(0)
        return np.exp(-d/(s*s))
    
    
    def train1(self, x, e = 0.5):
        diff = x - self.weights
        assert diff.shape == self.weights.shape
        summed = np.sum(np.square(diff), axis=2)
        # best matching unit
        bmu_x, bmu_y = np.unravel_index(np.argmin(summed), summed.shape)
        self.weights = self.weights + e * np.repeat(SelfOrganizingMap2D.change_function(bmu_x, bmu_y, self.xs, self.ys), self.shape[2]).reshape(self.shape) * diff
        
        self.bmu_x = bmu_x
        self.bmu_y = bmu_y
        
    def transform1(self, x):
        diff = x - self.weights
        assert diff.shape == self.weights.shape
        summed = np.sum(np.square(diff), axis=2)
        # best matching unit
        return np.unravel_index(np.argmin(summed), summed.shape)
    
class SelfOrganizingMap3D:
    '''
    This is a 3D cube map.
    '''
    def __init__(self, shape, mu, sigma, sigma_function=lambda x: 1, weights=None):
        assert len(shape) == 4
        self.shape = shape

        if weights is None:
            self.weights = np.random.normal(mu, sigma, shape)
        else:
            assert self.shape == weights.shape
            self.weights = weights

        nx, ny, nz = shape[0], shape[1], shape[2]
        self.xs = np.repeat(np.repeat(np.arange(nx), ny), nz)
        self.ys = np.tile(np.repeat(np.arange(ny), nz), nx)
        self.zs = np.tile(np.tile(np.arange(nz), ny), nx)
        
        self.sigma_function = sigma_function
    
    def distance_function(x0, y0, z0, x, y, z):
        dx, dy, dz = x0 - x, y0 - y, z0 - z
        return dx*dx + dy*dy + dz*dz

    def change_function(x0, y0, z0, x, y, z, sigma_function):
        d = SelfOrganizingMap3D.distance_function(x0, y0, z0, x, y, z)
        s = sigma_function(0)
        return np.exp(-d/(s*s))
    
    def train1(self, x, e = 0.5):
        diff = x - self.weights
        assert diff.shape == self.weights.shape
        summed = np.sum(np.square(diff), axis=3)
        assert len(summed.shape) == 3
        assert summed.shape[:3] == self.shape[:3]
        # best matching unit
        bmu_x, bmu_y, bmu_z = np.unravel_index(np.argmin(summed), summed.shape)
        self.weights = self.weights + e * np.repeat(SelfOrganizingMap3D.change_function(bmu_x, bmu_y, bmu_z, self.xs, self.ys, self.zs, self.sigma_function), self.shape[3]).reshape(self.shape) * diff
        
        self.bmu_x = bmu_x
        self.bmu_y = bmu_y
        self.bmu_z = bmu_z
        
    def transform1(self, x):
        diff = x - self.weights
        assert diff.shape == self.weights.shape
        summed = np.sum(np.square(diff), axis=3)
        assert len(summed.shape) == 3
        assert summed.shape[:3] == self.shape[:3]
        # best matching unit
        return np.unravel_index(np.argmin(summed), summed.shape)

def get_transformed_colors_3d(model, spikes):
    transformed = np.empty((spikes.shape[0], 3))
    colors = np.empty(spikes.shape[0], dtype='object')
    
    for i in range(spikes.shape[0]):
        x, y, z = model.transform1(spikes[i])
        transformed[i] = x, y, z
        colors[i] = get_color_3d(int(x/model.shape[0]*256), int(y/model.shape[1]*256), int(z/model.shape[2]*256))
        
    return transformed, colors
    
import torch

class Autoencoder3D:
    '''
    Has 3 hidden layers.
    '''
    class Autoencoder(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            super(Autoencoder3D.Autoencoder, self).__init__()
            self.fc1 = torch.nn.Linear(D_in, H)
            self.fc2 = torch.nn.LeakyReLU()
            self.fc3 = torch.nn.Linear(H, D_out)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

        def transform(self, x):
            with torch.no_grad():
                x = self.fc1(x)
                return x
            
    def __init__(self, length):
        self.network = Autoencoder3D.Autoencoder(length, 3, length)
    
    def trainBatch(self, spikes):
        x = torch.from_numpy(spikes)
        
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()
        
        epochs = 50
        batch_size = 512

        examples, _ = x.shape

        for e in range(epochs):
            for t in range(int(np.ceil(examples/batch_size))):
                x_slice = x[t * batch_size : (t+1) * batch_size]

                y_pred = self.network(x_slice)

                loss = loss_fn(y_pred, x_slice)
                if t % 100 == 99:
                    print(t, loss.item())

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
    def transformBatch(self, spikes):
        x = torch.from_numpy(spikes)
        return self.network.transform(x).numpy()

from sklearn.decomposition import PCA
    
class PrincipalComponentAnalysis3D():
    def __init__(self):
        self.model = PCA(n_components=3)
    
    def trainAll(self, spikes):
        self.model.fit(spikes)
    
    def transformBatch(self, spikes):
        return self.model.transform(spikes)
