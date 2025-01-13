/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {

    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 
    (a + bi)(c + di) = (ac - bd) + (ad + bc)i

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    uint thread_index;
    thread_index = threadIdx.x + blockDim.x * blockIdx.x;
    // TODO: potentially need to cast padded_length to a float here

    // First, elementwise multiplication
    while (thread_index < padded_length) {
        cufftComplex raw_data_i = raw_data[thread_index];
        cufftComplex impulse_i = impulse_v[thread_index];
        
        out_data[thread_index].x = (raw_data_i.x * impulse_i.x - raw_data_i.y * impulse_i.y) / padded_length;
        out_data[thread_index].y = (raw_data_i.x * impulse_i.y + raw_data_i.y * impulse_i.x) / padded_length;
        // out_data[thread_index].x = raw_data_i.x;
        // out_data[thread_index].y = raw_data_i.y;
        thread_index += blockDim.x * gridDim.x; // advance one grid (num of threads of grid) each time
    }

    // Normalizing afterwards （＾_＾）
    // 

}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */
   
    // "This is all wrong, scratch this"
    // extern declare share memory
    // divide input by number of threads in grid, so we have a non-consecutive chunk for each thread 
    // find max of each chunk
    // (Helena is not an english major ^ she is a ✨stem major✨)
    // write result back to index increment by length of array / num of blocks / 32 
    // syncthreads()
    // repeat until something is 1 (potentially nice to pad length of array to be power of 2)
    // if threadindex == 0: atomic max (collect max of each block)
    // sractch all of this, bad idea  :,<
    //:,< :,< :,< :,< :,< :,< :,< :,< :,< :,<


    // 1. Declaring shared memory: basically assume for this problem that
    // you can fit the entire array (divided by num_blocks).
    // 2. Copy global stuff to large shared memory
    // 3. Iteration for reduction, log(num_items_per_block) times: 
    //    - If thread index is > num_threads_per_block / (2 ** i), no-op
    //    - otherwise, move memory from 2*thread_idx and accumulate there
    //    - synchronize
    // 4. Copy stuff back

    // copy from out data global memory to shared_input
    // const uint length = padded_length;
    uint num_items_to_process_per_block = blockDim.x;
    // const uint nearest_pow_2 = 1u << (32 - __clz(num_items_to_process_per_block * 2 - 1));
;
    // __shared__ cufftComplex shm [nearest_pow_2];
    extern __shared__ cufftComplex shm [];
    uint thread_index;
    thread_index = threadIdx.x + blockDim.x * blockIdx.x; // Thread offset within the block?

    // Loop over grids
    while (thread_index < padded_length) {
        if (thread_index  + num_items_to_process_per_block >= num_items_to_process_per_block * 2) {
            shm[thread_index + num_items_to_process_per_block].x = 0;
        } else {
            shm[thread_index + num_items_to_process_per_block] = out_data[thread_index + num_items_to_process_per_block];
        }
        shm[thread_index] = out_data[thread_index];
    
        // Loop over the items within the block?
        while (num_items_to_process_per_block > 1) {
            float left_magnitude = abs(shm[thread_index].x);
            float right_magnitude = abs(shm[thread_index + num_items_to_process_per_block].x);
            float bigger = max(left_magnitude, right_magnitude);
            //  bigger = shm[thread_index], shm[thread_index + 32]; // stride 32 to avoid bank conflict
            shm[thread_index].x = bigger;
            num_items_to_process_per_block = num_items_to_process_per_block / 2;
            __syncthreads();
        }
        atomicMax(max_abs_val, shm[0].x);
        thread_index += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */
    uint thread_index;
    thread_index = threadIdx.x + blockDim.x * blockIdx.x; 
    while (thread_index < padded_length) {
        out_data[thread_index].x = out_data[thread_index].x / max_abs_val[0];
        thread_index += blockDim.x * gridDim.x;
    }

}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(
        raw_data, impulse_v, out_data, padded_length
    );
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock>>>(
        out_data, max_abs_val, padded_length
    );

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2:  Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(
        out_data, max_abs_val, padded_length
    );
}
