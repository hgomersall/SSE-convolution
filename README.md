SSE-convolution
===============

A demonstration of speeding up a 1D convolution using SSE

Information about the implementations is provided in convolve.c, which 
contains the interesting code.

Build it using `cmake .`

Test it with the python script `py_test_convolve.py`. This checks the output
and prints out the times taken for each implementation and the flops estimate.

The test_convolve c script seems to be broken (feel free to submit a patch).
