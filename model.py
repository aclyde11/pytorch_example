from torch import nn

class VAE(nn.Module):

    def __init__(self):
        super().__init__()


        self.linear1 = nn.Linear(1, 1, bias=True)
        self.activation = nn.Sigmoid()

        self.linear2 = nn.Linear(1, 1, bias=True)



    def forward(self, x):
        return self.linear1(x)


