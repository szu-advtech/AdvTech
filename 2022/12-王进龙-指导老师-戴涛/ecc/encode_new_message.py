#!/usr/bin/env python
# coding: utf-8

# In[1]:


import binascii
import numpy as np


# In[2]:


generator_matrix_file = "generator_matrices/generator_matrix_20190924.txt"
matrix = np.loadtxt(generator_matrix_file, dtype=str)
rows = []
for row in matrix:
    row_bits = []
    for bit in row:
        row_bits.append(float(bit))
    rows.append(row_bits)
matrix = np.array(rows)


# In[3]:


ascii_message = "MISL"
message = [float(x) for x in str(bin(int.from_bytes('MISL'.encode(), 'big'))) if x != "b"]


# In[4]:


codeword = np.matmul(np.transpose(message), matrix) % 2


# In[5]:


molbits_to_include = np.where(codeword == 1)[0]


# In[6]:


molbits_to_include


# In[7]:


print("".join([str(int(x)) for x in message]))
print("".join([str(int(x)) for x in codeword]))


# In[ ]:




