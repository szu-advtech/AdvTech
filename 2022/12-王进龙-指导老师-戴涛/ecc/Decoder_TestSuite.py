#!/usr/bin/env python
# coding: utf-8

# # Decoder Test Suite
# This Python 3 Jupyter notebook will test the accuracy and speed of the decoder. The decoder itself is written in C and must be compiled into and executable named `decoder`. These tests will automatically compile the program.

# ## Parameters
# Used for all tests. Random seeds are chose here explicitly so that tests are reproducible.

# In[14]:


import numpy as np
from subprocess import check_output, call
import os
import uuid
import json
import sys
import tempfile
import time
from importlib import reload
import encoder
encoder = reload(encoder)

k = 32 # Length of the message
n = 96 # Length of the codeword
seed = 5

def test_setup():
    # Set up a matrix and test messages
    np.random.seed(seed)
    matrix = np.concatenate(
        (np.identity(k, dtype=int), np.random.randint(2, size=(k, n-k))), axis=1)
    messages_to_test = [
        np.insert(np.zeros(k-1, dtype=int), 0, 1),
        np.append(np.zeros(k-1, dtype=int), 1),
        np.ones(k, dtype=int),
        np.array([0,1]*(int(k/2)) + [0]*(k%2)),
        np.array([1,0]*(int(k/2)) + [1]*(k%2)),
        np.array([0,1,1]*(int(k/3)) + [0]*(k%3)),
        np.array([1,1,0]*(int(k/3)) + [0]*(k%3))
    ]
    
    if(call(["make", "clean"]) != 0):
        print("Couldn't make clean: Cannot run tests.", file=sys.stderr)
    if(call(["make", "THREAD_POW=6"]) != 0):
        print("Couldn't make: cannot run tests", file=sys.stderr)
    return matrix, messages_to_test


# ## Correctness Test 1
# This test checks to make sure that the decoder will correctly decode some actual codewords. It would take too long to run through all codewords, so instead we test the all-zeroes codeword plus a small collection of others.

# In[15]:


matrix, messages_to_test = test_setup()
filename = str(uuid.uuid4()) + ".temp"
np.savetxt(filename, matrix, fmt="%d", delimiter="")
try:
    # And then test to make sure that the test messages can be correctly encoded and then decoded
    for message in messages_to_test:
        print("Testing message: " + "".join(map(str, message)))
        codeword = encoder.encode(message, matrix)
        codeword = "".join(map(str, codeword))
        print("Which corresponds to codeword: " + codeword)
        result = check_output(["./decoder", codeword, filename])
        result = json.loads(result.decode('utf-8'))
        if (result["message"] == "".join(map(str, message)) and result["distance"] == 0):
            print("Successfully decoded\n")
        else:
            print("Was not able to successfully decode. Received:", file=sys.stderr)
            print(result, file=sys.stderr)
finally:
    os.remove(filename)


# ## Correctness Test 2
# This test checks to make sure that the decoder will correctly decode some codewords *with accumulated errors*. Since it would take too long to run through all codewords and all errors, we test a small subset of codewords with varying levels of error corruption.

# In[16]:


# Compile the program
matrix, messages_to_test = test_setup()
# And then test the messages
def flip_bit(codeword, index):
    codeword[int(index)] = (codeword[int(index)]+1) % 2
filename = str(uuid.uuid4()) + ".temp"
np.savetxt(filename, matrix, fmt="%d", delimiter="")
try:
    for number_of_errors in [1, 5, 15]:
        print("Testing " + str(number_of_errors) + " error(s)")
        for message in messages_to_test:
            print("Testing message: " + "".join(map(str, message)))
            codeword = encoder.encode(message, matrix)
            for index in np.random.choice(range(96), size=number_of_errors, replace=False):
                flip_bit(codeword, index)
            codeword = ''.join(map(str, codeword))
            print("Which corresponds to codeword: " + codeword)                     
            result = check_output(["./decoder", codeword, filename])
            result = json.loads(result.decode('utf-8'))
            if (result["message"] == "".join(map(str, message)) and result["distance"] == number_of_errors):
                print("Successfully decoded\n")
            else:
                print("Was not able to successfully decode. Received:", file=sys.stderr)
                print(result, file=sys.stderr)
finally:
    os.remove(filename)


# ## Scaling Test
# Does not check for correctness

# In[17]:


matrix, messages_to_test = test_setup()
filename = str(uuid.uuid4()) + ".temp"
np.savetxt(filename, matrix, fmt="%d", delimiter="")
try:
    for scale in range(10):
        print("Assessing runtime of " + str(2**scale) + " threads.")
        if(call(["make", "clean"]) != 0):
            print("Couldn't make clean. Cannot run test.", file=sys.stderr)
        if(call(["make", "THREAD_POW="+str(scale)]) != 0):
            print("Couldn't make. Cannot run test.", file=sys.stderr)
        elapsed_times = []
        for message in messages_to_test:
            codeword = np.matmul(np.transpose(message), matrix, dtype=int) % 2
            start_time = time.time()
            result = check_output(["./decoder", "".join(map(str, codeword)), filename])
            elapsed_times.append(time.time() - start_time)
        print(elapsed_times)
finally:
    os.remove(filename)


# ## How much faster is decoding if we don't read the matrix from file?

# In[18]:


matrix, messages_to_test = test_setup()
print("Testing with reading the file")
filename = str(uuid.uuid4()) + ".temp"
np.savetxt(filename, matrix, fmt="%d", delimiter="")
try:
    elapsed_times = []
    for message in messages_to_test:
        codeword = np.matmul(np.transpose(message), matrix, dtype=int) % 2
        start_time = time.time()
        result = check_output(["./decoder", "".join(map(str, codeword)), filename])
        elapsed_times.append(time.time() - start_time)
    print(elapsed_times)
finally:
    os.remove(filename)
print("Testing without a file read")
elapsed_times = []
for message in messages_to_test:
    codeword = np.matmul(np.transpose(message), matrix, dtype=int) % 2
    start_time = time.time()
    result = check_output(["./decoder", "".join(map(str, codeword))])
    elapsed_times.append(time.time() - start_time)
print(elapsed_times)

