/* Copyright (C) 2012 Henry Gomersall <heng@cantab.net> 
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the organization nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY  THE AUTHOR ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 */

#ifndef _CONVOLVE_2D_H
#define _CONVOLVE_2D_H

#ifdef SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif

/* A macro that outputs a wrapper for each of the convolution routines.
 * The macro passed a name conv_func will output a function called
 * conv_func_multiple and it will have signature:
 * conv_func_multiple(float* in, float* out, int length,
 *                    float* kernel, int kernel_length, int N)
 * 
 * The additional N defines how many times to run the convolution function
 * conv_func(float* in, float* out, int length,
 *                    float* kernel, int kernel_length)
 * */

#ifndef MULTIPLE_CONVOLVE_2D
#define MULTIPLE_CONVOLVE_2D(FUNCTION_NAME) \
int FUNCTION_NAME ## _multiple(float* in, float* out, float* workspace, \
        int cols, int rows, float* kernel, int kernel_length, int N) \
{ \
    for(int i=0; i<N; i++){ \
        FUNCTION_NAME(in, out, workspace, cols, rows, kernel, \
                kernel_length); \
    } \
 \
    return 0; \
}
#endif

#ifdef SSE3

int convolve_sse_2d_separable(float* in, float* out, float* workspace,
        int cols, int rows, float* kernel, int kernel_length);
MULTIPLE_CONVOLVE_2D(convolve_sse_2d_separable);

#endif

#endif /*Header guard*/
