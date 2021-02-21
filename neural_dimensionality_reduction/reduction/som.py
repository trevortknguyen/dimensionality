import numpy as np

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

