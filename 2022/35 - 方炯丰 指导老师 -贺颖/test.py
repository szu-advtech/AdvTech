import torch
import numpy as np
#
# f = torch.load("preferred_states/reacher_easy_13.pt")
# print(f)
# # print(f.shape)
a = torch.randint(0,1,(2,2,2))
print(np.shape(a))
print(np.shape(a[:,:,None,:]))
print(a-a[:,:,None,:])