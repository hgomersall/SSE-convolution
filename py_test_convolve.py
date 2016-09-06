#!/usr/bin/env python

# Copyright (C) 2012 Henry Gomersall <heng@cantab.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the organization nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY  THE AUTHOR ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import ctypes
from pretty_print_times import pretty_print_times, colour

def check_convolution(input_array, test_output, kernel):

    correct_output = numpy.convolve(input_array, kernel, mode='valid')

    return numpy.allclose(correct_output, test_output, rtol=1e-4, atol=1e-5)

def get_function_wrapper(function_name, input_array, output_array, kernel,
        n_loops):

    if output_array.ndim != 1 or input_array.ndim != 1 or kernel.ndim != 1:
        raise ValueError('All the arrays should be dimension 1.')

    if len(kernel) > len(input_array):
        raise ValueError('The kernel should be shorter than the input '
                'array')

    if (input_array.dtype != 'float32' or output_array.dtype != 'float32'
            or kernel.dtype != 'float32'):
        raise ValueError('All the arrays should be of type \'float32\'')

    if len(output_array) != len(input_array) - len(kernel) + 1:
        raise ValueError('Output array should be of length '
                'len(input_array) - len(kernel) + 1')

    if len(input_array)%4 != 0:
        raise ValueError('The input array length should be divisible by 4.')

    if len(kernel)%4 != 0:
        raise ValueError('The kernel length should be divisible by 4.')

    lib = numpy.ctypeslib.load_library('libconvolve_funcs', '.')

    c_function = getattr(lib, function_name)
    c_function.restype = ctypes.c_int
    c_function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int]

    input_pointer = input_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
    output_pointer = output_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))

    length = len(input_array)
    kernel_pointer = kernel.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
    kernel_length = len(kernel)

    def function_wrapper():
        c_function(input_pointer, output_pointer, length, kernel_pointer,
                kernel_length, n_loops)

    return function_wrapper

timeit_vars = []
#lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
lengths = [256, 512, 1024, 2048, 4096, 8192]
functions = [
    'convolve_naive_multiple',
    'convolve_sse_simple_multiple',
    'convolve_sse_partial_unroll_multiple',
    'convolve_sse_in_aligned_multiple',
    'convolve_sse_in_aligned_fixed_kernel_multiple',
    'convolve_sse_unrolled_avx_vector_multiple',
    'convolve_sse_unrolled_vector_multiple',
    'convolve_avx_unrolled_vector_multiple',
    'convolve_avx_unrolled_vector_unaligned_multiple',
    'convolve_avx_unrolled_vector_unaligned_fma_multiple',
    'convolve_avx_unrolled_vector_m128_load_multiple',
    'convolve_avx_unrolled_vector_aligned_multiple',
    'convolve_avx_unrolled_vector_local_output_multiple',
    'convolve_avx_unrolled_vector_partial_aligned_multiple'
]

def time_convolutions():
    import timeit

    def make_setup_script(func):

        del timeit_vars[:]
        timeit_vars.append(input_array)
        timeit_vars.append(output_array)
        timeit_vars.append(kernel)
        timeit_vars.append(loops)

        script = 'import ' + __name__ + ' as module;'
        script += 'timeit_vars = module.timeit_vars;'
        script += 'function = module.get_function_wrapper(\'' + func
        script += '\', timeit_vars[0], timeit_vars[1], '
        script += 'timeit_vars[2], timeit_vars[3]);'

        return script


    kernel = numpy.float32(numpy.random.randn(16))
    times = numpy.zeros((len(functions), len(lengths)))
    flops = numpy.zeros((len(functions), len(lengths)))
    loops = 1000

    for k, each_length in enumerate(lengths):
        input_array = numpy.float32(numpy.random.randn(each_length))
        output_array = numpy.empty(
                len(input_array) - len(kernel) + 1, dtype='float32')

        for l, each_function in enumerate(functions):

            print(each_function, each_length)

            time = min(timeit.repeat('function()',
                setup=make_setup_script(each_function),
                repeat=20, number=1))

            print('valid:', check_convolution(input_array, output_array, kernel))

            # empty the output array
            output_array[:] = 0

            times[l, k] = time/loops
            flops[l, k] = (len(kernel) * len(output_array) * loops)/time

    return times, flops


if __name__ == '__main__':

    times, flops = time_convolutions()

    # Chop off each "convolve_"  and "_multiple" from each function name
    function_type = [each[9:-9] for each in functions]

    print(colour('\nTime in seconds\n', 'red'))
    pretty_print_times(times, lengths, function_type, highlight='min')

    print(colour('\nFlops\n', 'red'))
    pretty_print_times(flops, lengths, function_type, highlight='max')

