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


