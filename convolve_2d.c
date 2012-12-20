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

#include "convolve_2d.h"
#include <string.h>
#include <stdio.h>

#ifdef SSE3
#define KERNEL_LENGTH 16

static inline
int _convolve_along_rows_with_transpose(float* in, float* out, int cols,
        int rows, float* kernel)
{
    float kernel_block[4] __attribute__ ((aligned (16)));
    float in_aligned[4][cols] __attribute__ ((aligned (16)));

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

    for (int row=0; row<rows; row+=4){
        for (int sub_row=0; sub_row<4; sub_row++){

            float* in_row = in + (row+sub_row)*cols;
            float* out_row = out + (row+sub_row)*(cols-KERNEL_LENGTH+1);
            
            /* Create a set of 4 aligned arrays
             * Each array is offset by one sample from the one before
             */
            for(int i=0; i<4; i++){
                memcpy(in_aligned[i], (in_row+i), (cols-i)*sizeof(float));
            }

            for(int i=0; i<cols-KERNEL_LENGTH; i+=4){

                acc = _mm_setzero_ps();

                for(int k=0; k<KERNEL_LENGTH; k+=4){

                    int data_offset = i + k;

                    for (int l = 0; l < 4; l++){

                        data_block = _mm_load_ps(in_aligned[l] + data_offset);
                        prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                        acc = _mm_add_ps(acc, prod);
                    }
                }
                _mm_storeu_ps(out_row+i, acc);
                
                // Takes the transpose:
                //out[row + sub_row + i*rows] = acc[0];
                //out[row + sub_row + (i+1)*rows] = acc[1];
                //out[row + sub_row + (i+2)*rows] = acc[2];
                //out[row + sub_row + (i+3)*rows] = acc[3];

            }

            __m128 out1, out2, out3, out4;
            if (sub_row == 4){
                float* row1 = out_row - 3*(cols-KERNEL_LENGTH+1);
                float* row2 = out_row - 2*(cols-KERNEL_LENGTH+1);
                float* row3 = out_row - (cols-KERNEL_LENGTH+1);
                float* row4 = out_row;

                for(int i=0; i<cols-KERNEL_LENGTH; i+=4){
                    out1 = _mm_load_ps(row1 + i);
                    out2 = _mm_load_ps(row2 + i);
                    out3 = _mm_load_ps(row3 + i);
                    out4 = _mm_load_ps(row4 + i);
                    _MM_TRANSPOSE4_PS(out1, out2, out3, out4);
                
                    _mm_store_ps(row1 + i, out1);
                    _mm_store_ps(row2 + i, out2);
                    _mm_store_ps(row3 + i, out3);
                    _mm_store_ps(row4 + i, out4);
                }

            }

            // Need to do the last value as a special case
            int i = cols - KERNEL_LENGTH;
            
            //float* final_out = out_row + i;
            //float* final_out = out + row + i*rows; // takes the transpose
            float* final_out = out + row + sub_row + i*rows; // takes the transpose

            *final_out = 0.0;
            for(int k=0; k<KERNEL_LENGTH; k++){
                *final_out += in_aligned[0][i+k] * kernel[KERNEL_LENGTH - k - 1];
            }
        }
    }

    return 0;
}

int convolve_sse_2d_separable(float* in, float* out, float* workspace, 
        int cols, int rows, float* kernel, int kernel_length)
{
    float id_kernel[16];
    id_kernel[0] = 1.0;
    for(int i=1; i<16; i++){
        id_kernel[i] = 0;
    }

    _convolve_along_rows_with_transpose(in, workspace, cols, rows, kernel);
    _convolve_along_rows_with_transpose(workspace, out, rows, cols, id_kernel);

    return 0;
}

#endif
