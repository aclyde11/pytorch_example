from model import VAE
import numpy as np
from torch import optim
from torch.utils import data
from torch import nn
# return a single sample perfectly
class DataSet(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i,...], self.y[i,...]

def f(x):
    return 2   *x + 1


X = np.linspace(-1, 1, 1e5)
y = f(X)

X_test = np.linspace(-1, 10, 1e5)
y_test = f(X_test)


## NP Arrays

train_loader = data.DataLoader(dataset=DataSet(X, y), batch_size=32, shuffle=True, drop_last=True)
test_loader = data.DataLoader(dataset=DataSet(X_test, y), batch_size=32, shuffle=True, drop_last=True)


model = VAE()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

loss_function = nn.MSELoss(reduce='mean')



for epoch in range(10):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.float()
        y_batch = y_batch.float()

        x_batch = x_batch.view(32, 1)
        y_batch = y_batch.view(32, 1)


        optimizer.zero_grad()
        y_pred = model(x_batch)

        loss = loss_function(y_pred, y_batch)
        print(loss.item())
        loss.backward()
        optimizer.step()

