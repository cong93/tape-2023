import torch
import torch.nn as nn

class Learnable_Toeplitz_weight(nn.Module):
    def __init__(self, length_and_width,depth=1,channels=1):
        '''
        purpose:
            creates a learnable toeplitz tensor. by default will be a 2d tensor, shaped (n,n) where n=length_and_width
            if depth>1, then will have an additional dim of size 'depth' at the start of the shape
            if channels>1, then will have an additional dimension of size 'channels' at the end of the shape
        example use:
            #in the init function of the network:
            self.toeplitz=Learnable_Toeplitz_weight(10)
            self.toeplitz2=Learnable_toeplitz_weight(10, depth=3, channels=100)
            #in the forward function of the network:
            weight=self.toeplitz() #shape (10, 10)
            weight2=self.toeplitz2() #shape (3, 10, 10, 100)
        '''
        super(Learnable_Toeplitz_weight, self).__init__()
        n=length_and_width
        indices_tensor=torch.empty((n,n),dtype=torch.int32)
        i=0
        for start in range(n):
            for row, col in zip(range(start,n),range(0,n-start)):
                indices_tensor[row,col]=i
            i+=1
        for start in range(1,5):
            for row, col in zip(range(0,n-start),range(start,n)):
                indices_tensor[row,col]=i
            i+=1
        self.register_buffer('indices',indices_tensor)
        self.params = nn.Parameter(torch.randn((depth, int(2*n-1), channels)),requires_grad=True)
        self.depth_i=0 if depth==1 else slice(None)
        self.channels_i=0 if channels==1 else slice(None)
    def forward(self):
        return self.params[self.depth_i,self.indices,self.channels_i]
