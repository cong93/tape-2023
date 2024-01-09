implementations for:
  - learnable toeplitz tensor
  - embedding continuous variable

the learnable toeplitz tensor can be used like so as a bias for relative positions (1d positions based on token indices):
A=torch.matmul(q,k.transpose(-2,-1))*self.scale #(B,H,Nq,Nk)

bias=self.toeplitz() #(H, Nq, Nk)

A=softmax( A+bias.unsqueeze(0) ,dim=-1)

or like this (Your Transformer May Not be as Powerful as You Expect (2022)):
A=torch.matmul(q,k.transpose(-2,-1))*self.scale #(B,H,Nq,Nk)

C=self.toeplitz() #(H, Nq, Nk)

A=softmax(A, dim=-1) * C
