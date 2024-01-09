implementations for:
  - learnable toeplitz tensor (toeplitz.py)
  - continuous variable embedding (Continuous_Embedding.py)

learnable toeplitz tensor usage in "Your Transformer May Not be as Powerful as You Expect" (2022):

A=torch.matmul(q,k.transpose(-2,-1))*self.scale #(B,H,Nq,Nk)

B = torch.sum( q[:, :, :, None, :]*b[None, None, :, :, :] ,dim=-2)  #b shaped (N, N, C)

C=self.toeplitz() #(H, Nq, Nk) #make sure this is initialized as ones in the init function.

A=softmax(A+B, dim=-1) * C
