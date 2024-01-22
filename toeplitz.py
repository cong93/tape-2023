import torch
import torch.nn as nn

class Learnable_Toeplitz_weight(nn.Module):
    '''example use:
            self.toeplitz=Learnable_Toeplitz_weight(10)
            self.toeplitz2=Learnable_toeplitz_weight(10, depth=3, channels=100)
            weight=self.toeplitz() #shape (10, 10)
            weight2=self.toeplitz2() #shape (3, 10, 10, 100)'''
    def __init__(self, length_and_width,depth=1,channels=1,init_ones=False):
        super(Learnable_Toeplitz_weight, self).__init__()
        n=length_and_width
        indices_tensor=torch.zeros((n,n),dtype=torch.int32)
        for col,embd_i in zip(range(n-1,-1,-1),range(n)):
            steps=n-col
            for r,c in zip(range(steps),range(col,col+steps)):
                indices_tensor[r,c]=embd_i
        for row,embd_i in zip(range(1,n),range(n,int(n*2)-1)):
            steps=n-row
            for r,c in zip(range(row,row+steps),range(steps)):
                indices_tensor[r,c]=embd_i
        self.register_buffer('indices',indices_tensor)
        # print(indices_tensor)
        if init_ones:
            self.params = nn.Parameter(torch.ones((depth, int(2*n-1), channels)),requires_grad=True)
        else:
            self.params = nn.Parameter(torch.randn((depth, int(2*n-1), channels)), requires_grad=True)
        self.depth_i=0 if depth==1 else slice(None)
        self.channels_i=0 if channels==1 else slice(None)
    def forward(self):
        return self.params[self.depth_i,self.indices,self.channels_i]
