from model import *

a = torch.randn(25)

bla = mlp_vae(25,5,5,False,2)

print([i for i in bla.children()])

#print(bla(Variable(a)))
