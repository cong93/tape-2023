implementations for:
  - learnable toeplitz tensor
  - continuous variable embedding

learnable toeplitz tensor usage in "Your Transformer May Not be as Powerful as You Expect" (2022):

A=torch.matmul(q,k.transpose(-2,-1))*self.scale #(B,H,Nq,Nk)

C=self.toeplitz() #(H, Nq, Nk)

A=softmax(A, dim=-1) * C
