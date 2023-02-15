# Error Correcting Codes

## Quickstart
### To encode a message
There's a python module to help you encode a message called `encoder.py`. Encoding a message is a simple matrix-vector multiplication (mod 2).

### To decode a message
Compile the decoder with:

`$make clean`

`$make`

on the command line. This will create an executable called `decoder`. By default it will compile a threaded version and will run with 64 threads.

Run the decoder with:

`$decoder <data to decode>`

where `<data to decode>` is a 96 character string of 1s and 0s. The decoder has a built-in generator matrix that it will use for decoding. If you are using a different generator matrix then also pass the matric filename:

`$decoder <data to decode> <filename>`

where `<filename>` points to a plaintext file containing 32 lines, where each line contains a 96 character string of 1s and 0s. For an example look in `./generator_matrices`.

To make a version of the decoder that uses a different number of threads, do:

`make clean`

`make THREAD_POW=X`

where `X` is the log of the number of threads you want (number of threads must be a power of two, so simply specify 0 for 1 thread, 1 for 2 threads, 2 for 4 threads, etc).

## Test Suite
You can test the correctness and scalability of the decoder in the `Decoder_TestSuite.ipynb` Jupyter Notebook. You can open the notebook and select 'Run All'. If no red error text is printed then the decoder is working correctly.

## Error Correcting Code Background
Melissa should fill in some stuff here.

## How does `decoder.c` work?
This is a great question. Melissa should fill in this section.
