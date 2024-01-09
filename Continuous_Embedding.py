import torch
import torch.nn as nn

class Continuous_Embedding(nn.Module):
    def __init__(self, min_val, max_val, out_dim, embd_dim=None,
                 n_boundaries=50, boundary_spacing='linear'):
        super(Continuous_Embedding, self).__init__()
        '''purpose:
            - maps a tensor, X, shaped (B, N, 1) of float dtype, to tensor of learned values (B, N, out_dim).
            - values that are below or above min_val and max_val respectively are just mapped to the vectors for
              exactly min_val and max_val.
            - the forward function also has the option to turn x into relative values (B, N, N, 1)
              and map to learned vectors representing relative differences of X
        parameters:
            min_val: minimum value of the variable
            max_val: maximum value of the variable
            out_dim: size of dim=-1 for the output tensor (B, N, out_dim)
            embd_dim: size learned embedding, by default will be equal to out_dim, but it can be set to lower
            n_boundaries: number of vectors representing discrete values between min_val and max_val
            boundary_spacing:
            - 'linear' (linear spacing), example given min_val=0, max_val=1, n_boundaries=5: 0., 0.25, 0.50, 0.75, 1.
            - 'linear2', linear spacing, but also extends to the left. example: -1., -0.75, -0.50, -0.25, 0., 0.25, 0.50, 0.75, 1.
            - 'exp' (exponential spacing)
            - 'exp2' similar to linear2 but with exponential spacing.
            **for linear 2 and exp2, they will have double number of learned vectors as n_boundaries, also min_val 
            should be the middle value in the spacing, eg., 0 instead of -1)
        example use:
            #in the init function:
            self.relative_time_embd = Continuous_Embedding(0,1,50,boundary_spacing='exp2')
            self.time_embd = Continuous_Embedding(0,1,20,boundary_spacing='linear')
            #in the forward function:
            X,time=torch.split(X,X.shape[-1]-1,dim=-1) #assuming X is a irregular time series (B,N,C) with time being [:,:,-1]
            time = self.time_embd(time) #B, N, 20
            rel_time = self.relative_time_embd(time,relative=True) #B, N, N, 50
        '''
        if boundary_spacing=='linear':
            boundaries=torch.linspace(min_val, max_val, n_boundaries)
            self.embd_length= n_boundaries + 1
        elif boundary_spacing=='linear2':
            boundaries=torch.linspace(0, 1, n_boundaries)
            boundaries=torch.cat((-torch.flip(boundaries,dims=[0])[:-1],boundaries),dim=0)
            boundaries=boundaries*(max_val-min_val)+min_val
            self.embd_length= n_boundaries * 2
        elif boundary_spacing=='exp':
            boundaries=self.expspace(min_val, max_val, length=n_boundaries, mirror=False,
                                     temperature=2)
            self.embd_length= n_boundaries + 1
        else: #self.spacing=='exp2':
            boundaries=self.expspace(min_val, max_val, length=n_boundaries, mirror=True,
                                     temperature=2)
            self.embd_length= n_boundaries * 2
        self.register_buffer('boundaries',boundaries)

        self.out_dim=out_dim
        if embd_dim==None:embd_dim=out_dim
        self.embd_dim=embd_dim
        self.embd=nn.Parameter(torch.randn((self.embd_length, embd_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.embd)
        self.embd_out_of_bounds=nn.Parameter(torch.randn((2, out_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.embd_out_of_bounds)
        self.mlp=nn.Sequential(
            nn.Linear(int(embd_dim*2+1),int(2*embd_dim+1)),
            nn.GELU(),
            nn.Linear(int(2*embd_dim+1),out_dim)
        )

    @staticmethod
    def expspace(min_val, max_val, length, mirror, temperature):
        linear_values = torch.linspace(0,1, length)
        e = torch.e**(linear_values*temperature)
        e=(e-torch.min(e))/(torch.max(e)-torch.min(e))
        if mirror:
            e2=-torch.flip(e,dims=[0])[:-1]
            e=torch.cat((e2,e),axis=0)
        scaled_e = e * (max_val - min_val) + min_val
        return scaled_e

    def forward(self,x,x2=None,relative=False):
        #x: B,N,1, x1: B N 1
        if relative:
            if x2==None:
                x=x[:,:,None,:]-x[:,None,:,:]
            else:
                x=x[:,:,None,:]-x2[:,None,:,:]
            b,nq,nk,c=x.shape
            x=x.reshape(x.shape[0],nq*nk,x.shape[3])
        assert x.shape[-1]==1 and x.ndim==3
        output=torch.empty((x.shape[0],x.shape[1],self.out_dim),dtype=torch.float32,device=x.device)
        boundaries=self.boundaries
        indices=torch.searchsorted(boundaries,x.contiguous())

        boundaries=torch.cat((boundaries[0:1],boundaries,boundaries[-1:]),dim=0)
        lower_boundaries=boundaries[indices]
        higher_boundaries=boundaries[indices+1]
        ranges_=higher_boundaries-lower_boundaries
        distances=(x-lower_boundaries)/ranges_

        mask=torch.argwhere((indices>0) & (indices<self.embd_length-1))
        if mask.shape[0]!=x.numel():
            mask2=torch.argwhere(indices==0)
            mask3=torch.argwhere(indices==self.embd_length-1)
            if mask2.shape[0]>0:
                output[mask2[:,0],mask2[:,1],:]=self.embd_out_of_bounds[0]
            if mask3.shape[0]>0:
                output[mask3[:,0],mask3[:,1],:]=self.embd_out_of_bounds[1]

        indices=indices[mask[:,0],mask[:,1]].squeeze(-1)
        distances=distances[mask[:,0],mask[:,1]]

        embd=torch.cat((self.embd[indices], self.embd[indices+1], distances),dim=-1)
        # out=self.mlp(embd)
        out=self.mlp(embd)
        output[mask[:,0],mask[:,1],:]=out
        if relative:
            output=output.reshape( b,nq,nk,self.out_dim )
        return output #B,N,embd_dim
