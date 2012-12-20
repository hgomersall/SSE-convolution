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

#include "convolve.h"
#include <string.h>
#include <stdio.h>

/* A set of convolution routines, all of which present the same interface
 * (albeit, with some of them having restrictions on the size and shape of 
 * the input data)
 *
 * ``out'' is filled with the same values as would be returned by
 * numpy.convolve(in, kernel, mode='valid').
 *
 * All the convolve functions have the same signature and interface.
 *
 * */


/* A simple implementation of a 1D convolution that just iterates over
 * scalar values of the input array. 
 *
 * Returns the same as numpy.convolve(in, kernel, mode='valid')
 * */
int convolve_naive(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    for(int i=0; i<=length-kernel_length; i++){

        out[i] = 0.0;
        for(int k=0; k<kernel_length; k++){
            out[i] += in[i+k] * kernel[kernel_length - k - 1];
        }
    }

    return 0;
}


#ifdef SSE3


/* Vectorize the algorithm to compute 4 output samples in parallel.
 *
 * Each kernel value is repeated 4 times, which can then be used on
 * 4 input samples in parallel. Stepping over these as in naive
 * means that we get 4 output samples for each inner kernel loop.
 *
 * For this, we need to pre-reverse the kernel, rather than doing
 * the loopup each time in the inner loop.
 *
 * The last value needs to be done as a special case.
 */
int convolve_sse_simple(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[4] __attribute__ ((aligned (16)));

    __m128 kernel_reverse[kernel_length] __attribute__ ((aligned (16)));    
    __m128 data_block __attribute__ ((aligned (16)));

    __m128 prod __attribute__ ((aligned (16)));
    __m128 acc __attribute__ ((aligned (16)));

    // Reverse the kernel and repeat each value across a 4-vector
    for(int i=0; i<kernel_length; i++){
        kernel_block[0] = kernel[kernel_length - i - 1];
        kernel_block[1] = kernel[kernel_length - i - 1];
        kernel_block[2] = kernel[kernel_length - i - 1];
        kernel_block[3] = kernel[kernel_length - i - 1];

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    for(int i=0; i<length-kernel_length; i+=4){

        // Zero the accumulator
        acc = _mm_setzero_ps();

        /* After this loop, we have computed 4 output samples
         * for the price of one.
         * */
        for(int k=0; k<kernel_length; k++){

            // Load 4-float data block. These needs to be an unaliged
            // load (_mm_loadu_ps) as we step one sample at a time.
            data_block = _mm_loadu_ps(in + i + k);
            prod = _mm_mul_ps(kernel_reverse[k], data_block);

            // Accumulate the 4 parallel values
            acc = _mm_add_ps(acc, prod);
        }
        _mm_storeu_ps(out+i, acc);

    }

    // Need to do the last value as a special case
    int i = length - kernel_length;
    out[i] = 0.0;
    for(int k=0; k<kernel_length; k++){
        out[i] += in[i+k] * kernel[kernel_length - k - 1];
    }

    return 0;
}


/* As convolve_sse_simple plus...
 *
 * We specify that the kernel must have a length which is a multiple
 * of 4. This allows us to define a fixed inner-most loop that can be 
 * unrolled by the compiler
 */
int convolve_sse_partial_unroll(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[4] __attribute__ ((aligned (16)));

    __m128 kernel_reverse[kernel_length] __attribute__ ((aligned (16)));    
    __m128 data_block __attribute__ ((aligned (16)));

    __m128 prod __attribute__ ((aligned (16)));
    __m128 acc __attribute__ ((aligned (16)));

    // Repeat the kernel across the vector
    for(int i=0; i<kernel_length; i++){
        kernel_block[0] = kernel[kernel_length - i - 1];
        kernel_block[1] = kernel[kernel_length - i - 1];
        kernel_block[2] = kernel[kernel_length - i - 1];
        kernel_block[3] = kernel[kernel_length - i - 1];

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }
    
    for(int i=0; i<length-kernel_length; i+=4){

        acc = _mm_setzero_ps();

        for(int k=0; k<kernel_length; k+=4){

            int data_offset = i + k;

            for (int l = 0; l < 4; l++){

                data_block = _mm_loadu_ps(in + data_offset + l);
                prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                acc = _mm_add_ps(acc, prod);
            }
        }
        _mm_storeu_ps(out+i, acc);

    }

    // Need to do the last value as a special case
    int i = length - kernel_length;
    out[i] = 0.0;
    for(int k=0; k<kernel_length; k++){
        out[i] += in[i+k] * kernel[kernel_length - k - 1];
    }

    return 0;
}


/* As convolve_sse_partial_unroll plus...
 *
 * We repeat the input data 4 times, with each repeat being shifted
 * by one sample from the previous repeat:
 * original: [0, 1, 2, 3, 4, 5, ...]
 *
 * repeat 1: [0, 1, 2, 3, 4, 5, ...]
 * repeat 2: [1, 2, 3, 4, 5, 6, ...]
 * repeat 3: [2, 3, 4, 5, 6, 7, ...]
 * repeat 4: [3, 4, 5, 6, 7, 8, ...]
 *
 * The effect of this is to create a set of arrays that encapsulate
 * a 16-byte alignment for every possible offset within the data.
 * Sample 0 is aligned in repeat 1, Sample 1 is aligned in repeat 1
 * etc. We then wrap around and sample 4 is aligned on repeat 1.
 *
 * The copies can be done fast with a memcpy.
 *
 * This means that in our unrolled inner-most loop, we can now do
 * an aligned data load (_mm_load_ps), speeding up the algorithm 
 * by ~2x.
 * */
int convolve_sse_in_aligned(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[4] __attribute__ ((aligned (16)));
    float in_aligned[4][length] __attribute__ ((aligned (16)));

    __m128 kernel_reverse[kernel_length] __attribute__ ((aligned (16)));    
    __m128 data_block __attribute__ ((aligned (16)));

    __m128 prod __attribute__ ((aligned (16)));
    __m128 acc __attribute__ ((aligned (16)));

    // Repeat the kernel across the vector
    for(int i=0; i<kernel_length; i++){
        kernel_block[0] = kernel[kernel_length - i - 1];
        kernel_block[1] = kernel[kernel_length - i - 1];
        kernel_block[2] = kernel[kernel_length - i - 1];
        kernel_block[3] = kernel[kernel_length - i - 1];

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    /* Create a set of 4 aligned arrays
     * Each array is offset by one sample from the one before
     */
    for(int i=0; i<4; i++){
        memcpy(in_aligned[i], (in+i), (length-i)*sizeof(float));
    }

    for(int i=0; i<length-kernel_length; i+=4){

        acc = _mm_setzero_ps();

        for(int k=0; k<kernel_length; k+=4){

            int data_offset = i + k;

            for (int l = 0; l < 4; l++){

                data_block = _mm_load_ps(in_aligned[l] + data_offset);
                prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                acc = _mm_add_ps(acc, prod);
            }
        }
        _mm_storeu_ps(out+i, acc);

    }

    // Need to do the last value as a special case
    int i = length - kernel_length;
    out[i] = 0.0;
    for(int k=0; k<kernel_length; k++){
        out[i] += in_aligned[0][i+k] * kernel[kernel_length - k - 1];
    }

    return 0;
}

/* In this case, the kernel is assumed to be a fixed length, this
 * allows the compiler to do another level of loop unrolling.
 */
#define KERNEL_LENGTH 16
int convolve_sse_in_aligned_fixed_kernel(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[4] __attribute__ ((aligned (16)));
    float in_aligned[4][length] __attribute__ ((aligned (16)));

    __m128 kernel_reverse[KERNEL_LENGTH] __attribute__ ((aligned (16)));    
    __m128 data_block __attribute__ ((aligned (16)));

    __m128 prod __attribute__ ((aligned (16)));
    __m128 acc __attribute__ ((aligned (16)));

    // Repeat the kernel across the vector
    for(int i=0; i<KERNEL_LENGTH; i++){
        kernel_block[0] = kernel[KERNEL_LENGTH - i - 1];
        kernel_block[1] = kernel[KERNEL_LENGTH - i - 1];
        kernel_block[2] = kernel[KERNEL_LENGTH - i - 1];
        kernel_block[3] = kernel[KERNEL_LENGTH - i - 1];

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    /* Create a set of 4 aligned arrays
     * Each array is offset by one sample from the one before
     */
    for(int i=0; i<4; i++){
        memcpy(in_aligned[i], (in+i), (length-i)*sizeof(float));
    }

    for(int i=0; i<length-KERNEL_LENGTH; i+=4){

        acc = _mm_setzero_ps();

        for(int k=0; k<KERNEL_LENGTH; k+=4){

            int data_offset = i + k;

            for (int l = 0; l < 4; l++){

                data_block = _mm_load_ps(in_aligned[l] + data_offset);
                prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                acc = _mm_add_ps(acc, prod);
            }
        }
        _mm_storeu_ps(out+i, acc);

    }

    // Need to do the last value as a special case
    int i = length - KERNEL_LENGTH;
    out[i] = 0.0;
    for(int k=0; k<KERNEL_LENGTH; k++){
        out[i] += in_aligned[0][i+k] * kernel[KERNEL_LENGTH - k - 1];
    }

    return 0;
}

/* As convolve_sse_in_aligned_fixed_kernel but with AVX instructions
 * emulated with SSE.
 * */
#define KERNEL_LENGTH 16
#define ALIGNMENT 32
#define SSE_SIMD_LENGTH 4
#define AVX_SIMD_LENGTH 8
int convolve_sse_unrolled_avx_vector(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[SSE_SIMD_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));
    float in_aligned[SSE_SIMD_LENGTH][length] __attribute__ (
            (aligned (ALIGNMENT)));

    __m128 kernel_reverse[KERNEL_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));    
    __m128 data_block __attribute__ ((aligned (ALIGNMENT)));

    __m128 prod __attribute__ ((aligned (ALIGNMENT)));
    __m128 acc0 __attribute__ ((aligned (ALIGNMENT)));
    __m128 acc1 __attribute__ ((aligned (ALIGNMENT)));

    // Repeat the kernel across the vector
    for(int i=0; i<KERNEL_LENGTH; i++){
        for(int j=0; j<SSE_SIMD_LENGTH; j++){
            kernel_block[j] = kernel[KERNEL_LENGTH - i - 1];
        }

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    /* Create a set of FLOATS_PER_SIMD_LENGTH aligned arrays
     * Each array is offset by one sample from the one before
     */
    for(int i=0; i<4; i++){
        memcpy(in_aligned[i], (in+i), (length-i)*sizeof(float));
    }

    for(int i=0; i<length-KERNEL_LENGTH; i+=AVX_SIMD_LENGTH){

        acc0 = _mm_setzero_ps();
        acc1 = _mm_setzero_ps();

        for(int k=0; k<KERNEL_LENGTH; k+=AVX_SIMD_LENGTH){

            int data_offset = i + k;

            for (int l = 0; l < 4; l++){

                data_block = _mm_load_ps(in_aligned[l] + data_offset);
                prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                acc0 = _mm_add_ps(acc0, prod);

                data_block = _mm_load_ps(in_aligned[l] + data_offset + 4);
                prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                acc1 = _mm_add_ps(acc1, prod);

                data_block = _mm_load_ps(in_aligned[l] + data_offset + 4);
                prod = _mm_mul_ps(kernel_reverse[k+l+4], data_block);

                acc0 = _mm_add_ps(acc0, prod);

                data_block = _mm_load_ps(in_aligned[l] + data_offset + 8);
                prod = _mm_mul_ps(kernel_reverse[k+l+4], data_block);

                acc1 = _mm_add_ps(acc1, prod);
            }
        }
        _mm_storeu_ps(out+i, acc0);
        _mm_storeu_ps(out+i+4, acc1);

    }

    // Need to do the last value as a special case
    int i = length - KERNEL_LENGTH;
    out[i] = 0.0;
    for(int k=0; k<KERNEL_LENGTH; k++){
        out[i] += in_aligned[0][i+k] * kernel[KERNEL_LENGTH - k - 1];
    }

    return 0;
}

#define VECTOR_LENGTH 16
int convolve_sse_unrolled_vector(float* in, float* out, 
        int length, float* kernel, int kernel_length)
{
    float kernel_block[SSE_SIMD_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));
    float in_aligned[SSE_SIMD_LENGTH][length] __attribute__ (
            (aligned (ALIGNMENT)));

    __m128 kernel_reverse[KERNEL_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));    
    __m128 data_block __attribute__ ((aligned (ALIGNMENT)));

    __m128 prod __attribute__ ((aligned (ALIGNMENT)));
    __m128 acc0 __attribute__ ((aligned (ALIGNMENT)));
    __m128 acc1 __attribute__ ((aligned (ALIGNMENT)));
    __m128 acc2 __attribute__ ((aligned (ALIGNMENT)));
    __m128 acc3 __attribute__ ((aligned (ALIGNMENT)));

    // Repeat the kernel across the vector
    for(int i=0; i<KERNEL_LENGTH; i++){
        for(int j=0; j<SSE_SIMD_LENGTH; j++){
            kernel_block[j] = kernel[KERNEL_LENGTH - i - 1];
        }

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    /* Create a set of 4 aligned arrays
     * Each array is offset by one sample from the one before
     */
    for(int i=0; i < SSE_SIMD_LENGTH; i++){
        memcpy(in_aligned[i], (in+i), (length-i)*sizeof(float));
    }

    for(int i=0; i<length-KERNEL_LENGTH; i+=VECTOR_LENGTH){

        acc0 = _mm_setzero_ps();
        acc1 = _mm_setzero_ps();
        acc2 = _mm_setzero_ps();
        acc3 = _mm_setzero_ps();

        for(int k=0; k<KERNEL_LENGTH; k+=VECTOR_LENGTH){

            int data_offset = i + k;

            for (int l = 0; l < SSE_SIMD_LENGTH; l++){

                for (int m = 0; m < VECTOR_LENGTH; m+=SSE_SIMD_LENGTH) {

                    data_block = _mm_load_ps(
                            in_aligned[l] + data_offset + m);
                    prod = _mm_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc0 = _mm_add_ps(acc0, prod);

                    data_block = _mm_load_ps(in_aligned[l] + data_offset 
                            + m + SSE_SIMD_LENGTH);
                    prod = _mm_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc1 = _mm_add_ps(acc1, prod);

                    data_block = _mm_load_ps(in_aligned[l] + data_offset 
                            + m + SSE_SIMD_LENGTH * 2);
                    prod = _mm_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc2 = _mm_add_ps(acc2, prod);

                    data_block = _mm_load_ps(in_aligned[l] + data_offset 
                            + m + SSE_SIMD_LENGTH * 3);
                    prod = _mm_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc3 = _mm_add_ps(acc3, prod);
                }
            }
        }
        _mm_storeu_ps(out+i, acc0);
        _mm_storeu_ps(out+i+SSE_SIMD_LENGTH, acc1);
        _mm_storeu_ps(out+i+SSE_SIMD_LENGTH*2, acc2);
        _mm_storeu_ps(out+i+SSE_SIMD_LENGTH*3, acc3);

    }

    // Need to do the last value as a special case
    int i = length - KERNEL_LENGTH;
    out[i] = 0.0;
    for(int k=0; k<KERNEL_LENGTH; k++){
        out[i] += in_aligned[0][i+k] * kernel[KERNEL_LENGTH - k - 1];
    }

    return 0;
}


#endif

#ifdef AVX

int convolve_avx_unrolled_vector(float* in, float* out, 
        int length, float* kernel, int kernel_length)
{
    float kernel_block[AVX_SIMD_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));
    float in_aligned[AVX_SIMD_LENGTH][length] __attribute__ (
            (aligned (ALIGNMENT)));

    __m256 kernel_reverse[KERNEL_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));    
    __m256 data_block __attribute__ ((aligned (ALIGNMENT)));

    __m256 prod __attribute__ ((aligned (ALIGNMENT)));
    __m256 acc0 __attribute__ ((aligned (ALIGNMENT)));
    __m256 acc1 __attribute__ ((aligned (ALIGNMENT)));

    // Repeat the kernel across the vector
    for(int i=0; i<KERNEL_LENGTH; i++){
        for(int j=0; j<AVX_SIMD_LENGTH; j++){
            kernel_block[j] = kernel[KERNEL_LENGTH - i - 1];
        }

        kernel_reverse[i] = _mm256_load_ps(kernel_block);
    }

    /* Create a set of 4 aligned arrays
     * Each array is offset by one sample from the one before
     */
    for(int i=0; i < SSE_SIMD_LENGTH; i++){
        memcpy(in_aligned[i], (in+i), (length-i)*sizeof(float));
    }

    for(int i=0; i<length-KERNEL_LENGTH; i+=VECTOR_LENGTH){

        acc0 = _mm256_setzero_ps();
        acc1 = _mm256_setzero_ps();

        for(int k=0; k<KERNEL_LENGTH; k+=VECTOR_LENGTH){

            int data_offset = i + k;

            for (int l = 0; l < SSE_SIMD_LENGTH; l++){

                for (int m = 0; m < VECTOR_LENGTH; m+=SSE_SIMD_LENGTH) {

                    data_block = _mm256_loadu_ps(
                            in_aligned[l] + data_offset + m);
                    prod = _mm256_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc0 = _mm256_add_ps(acc0, prod);

                    data_block = _mm256_loadu_ps(in_aligned[l] + data_offset 
                            + m + AVX_SIMD_LENGTH);
                    prod = _mm256_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc1 = _mm256_add_ps(acc1, prod);

                }
            }
        }
        _mm256_storeu_ps(out+i, acc0);
        _mm256_storeu_ps(out+i+AVX_SIMD_LENGTH, acc1);

    }

    // Need to do the last value as a special case
    int i = length - KERNEL_LENGTH;
    out[i] = 0.0;
    for(int k=0; k<KERNEL_LENGTH; k++){
        out[i] += in_aligned[0][i+k] * kernel[KERNEL_LENGTH - k - 1];
    }

    return 0;
}

int convolve_avx_unrolled_vector_partial_aligned(float* in, float* out, 
        int length, float* kernel, int kernel_length)
{
    float kernel_block[AVX_SIMD_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));
    float in_aligned[AVX_SIMD_LENGTH][length] __attribute__ (
            (aligned (ALIGNMENT)));

    __m256 kernel_reverse[KERNEL_LENGTH] __attribute__ (
            (aligned (ALIGNMENT)));    
    __m256 data_block __attribute__ ((aligned (ALIGNMENT)));

    __m256 prod __attribute__ ((aligned (ALIGNMENT)));
    __m256 acc0 __attribute__ ((aligned (ALIGNMENT)));
    __m256 acc1 __attribute__ ((aligned (ALIGNMENT)));

    // Repeat the kernel across the vector
    for(int i=0; i<KERNEL_LENGTH; i++){
        for(int j=0; j<AVX_SIMD_LENGTH; j++){
            kernel_block[j] = kernel[KERNEL_LENGTH - i - 1];
        }

        kernel_reverse[i] = _mm256_load_ps(kernel_block);
    }

    /* Create a set of SSE_SIMD_LENGTH aligned arrays
     * Each array is offset by one sample from the one before
     */
    for(int i=0; i < SSE_SIMD_LENGTH; i++){
        memcpy(in_aligned[i], (in+i), (length-i)*sizeof(float));
    }

    for(int i=0; i<length-KERNEL_LENGTH; i+=VECTOR_LENGTH){

        acc0 = _mm256_setzero_ps();
        acc1 = _mm256_setzero_ps();

        for(int k=0; k<KERNEL_LENGTH; k+=VECTOR_LENGTH){

            int data_offset = i + k;

            for (int l = 0; l < SSE_SIMD_LENGTH; l++){

                for (int m = 0; m < VECTOR_LENGTH; m+=AVX_SIMD_LENGTH) {

                    data_block = _mm256_load_ps(
                            in_aligned[l] + data_offset + m);
                    prod = _mm256_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc0 = _mm256_add_ps(acc0, prod);

                    data_block = _mm256_load_ps(in_aligned[l] + data_offset 
                            + m + AVX_SIMD_LENGTH);
                    prod = _mm256_mul_ps(kernel_reverse[k+l+m], data_block);

                    acc1 = _mm256_add_ps(acc1, prod);

                    data_block = _mm256_loadu_ps(
                            in_aligned[l] + data_offset + m + SSE_SIMD_LENGTH);
                    prod = _mm256_mul_ps(
                            kernel_reverse[k+l+m+SSE_SIMD_LENGTH], data_block);

                    acc0 = _mm256_add_ps(acc0, prod);

                    data_block = _mm256_loadu_ps(in_aligned[l] + data_offset 
                            + m + SSE_SIMD_LENGTH + AVX_SIMD_LENGTH);
                    prod = _mm256_mul_ps(
                            kernel_reverse[k+l+m+SSE_SIMD_LENGTH], data_block);

                    acc1 = _mm256_add_ps(acc1, prod);


                }
            }
        }
        _mm256_storeu_ps(out+i, acc0);
        _mm256_storeu_ps(out+i+AVX_SIMD_LENGTH, acc1);

    }

    // Need to do the last value as a special case
    int i = length - KERNEL_LENGTH;
    out[i] = 0.0;
    for(int k=0; k<KERNEL_LENGTH; k++){
        out[i] += in_aligned[0][i+k] * kernel[KERNEL_LENGTH - k - 1];
    }

    return 0;
}


#endif
