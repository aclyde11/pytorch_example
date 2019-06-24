from model import VAE
import numpy as np
from torch import optim
from torch.utils import data
from torch import nn
import torch
from tqdm import tqdm
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

train_loader = data.DataLoader(dataset=DataSet(X, y), batch_size=32, shuffle=True, drop_last=True, num_workers=2)
test_loader = data.DataLoader(dataset=DataSet(X_test, y), batch_size=32, shuffle=True, drop_last=True, num_workers=2)




model = VAE()
optimizer = optim.SGD(model.parameters(), lr=0.0001) #used to optimize model

loss_function = nn.MSELoss(reduce='mean') #reduce means how you combine the loss across the batch



for epoch in range(10):
    ##fancy way to show off training:

    tqdm_data = tqdm(train_loader,
                     desc='Training (epoch #{})'.format(epoch))
    model.train()
    #training loop
    for i, (x_batch, y_batch) in enumerate(tqdm_data):
        x_batch = x_batch.float() #convert data to FP32
        y_batch = y_batch.float()

        x_batch = x_batch.view(32, 1) # only needed because the input is of size 1, normally not needed
        y_batch = y_batch.view(32, 1) # only needed because the input is of size 1, normally not needed


        optimizer.zero_grad() # Need to clear out gradients before computing on batch!
        y_pred = model(x_batch) #run data through model

        loss = loss_function(y_pred, y_batch) #compute loss
        loss.backward() #compute gradients
        optimizer.step() #take a step of gradients * lr

        loss_value = loss.item()
        postfix = [f'loss={loss_value:.5f}']
        tqdm_data.set_postfix_str(' '.join(postfix))

    tqdm_data = tqdm(train_loader,
                     desc='Validation (epoch #{})'.format(epoch))
    #validation loop
    model.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(tqdm_data):
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            x_batch = x_batch.view(32, 1) # only needed because the input is of size 1, normally not needed
            y_batch = y_batch.view(32, 1) # only needed because the input is of size 1, normally not needed


            optimizer.zero_grad() # Need to clear out gradients before computing on batch!
            y_pred = model(x_batch) #run data through model

            loss = loss_function(y_pred, y_batch) #compute loss

            loss_value = loss.item()
            postfix = [f'loss={loss_value:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))
