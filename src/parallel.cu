#include "parallel.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void initialize(u32 *d_out, u32 value){
    u32 id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > MAX_BINS - 1) return;
    d_out[id] = value;
}

__global__ void m_hist(u8 *d_in, u32 *d_out, u32 length) {
    u32 id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > length - 1) return;
   /* 
    //__shared__ bool bin_mat[MAX_BINS * MAX_THREADS];
    __shared__ u8 bin_arr[MAX_THREADS];
    
    //TODO: Check if bool is initialized by default to false
    //bin[threadIdx.x] = 0;

    //__syncthreads();

    u8 bin_index = d_in[id];
    //u32 offset = threadIdx.x * MAX_BINS + bin_index;
    //bin_mat[offset] = true;
    bin_arr[threadIdx.x] = bin_index;

    __syncthreads();

    __shared__ u32 bin[MAX_BINS];
    if (threadIdx.x < MAX_BINS) {
        //TODO: Initialization check
        bin[threadIdx.x] = 0;
        for(int i = 0; i < MAX_THREADS; i++) {
           if(bin_arr[i] == threadIdx.x) bin[threadIdx.x]++; 
        }
        if(bin[threadIdx.x] != 0) atomicAdd(&d_out[threadIdx.x], bin[threadIdx.x]);
    }*/
    u32 bin = int(d_in[id]);
    atomicAdd(&d_out[bin], 1);

}

int histogram(u8 *h_in, u32 *h_out, u32 length) {
    for (int i = 0; i < length; i++) {
        h_out[h_in[i]]++;
    }
    return 0;
}

int m_histogram(u8 *h_in, u32 *h_out, u32 length) {
    /**
     * Allocate memory for the bins [0-255]
     */
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    u32 *d_out;
    cudaMalloc((void **) &d_out, MAX_BINS * sizeof(u32));
    /**
     * Call kernel initialize() to initialize all values to 0.
     */
    dim3 grid = dim3((int) ceil((float) MAX_BINS/MAX_THREADS));
    dim3 block = dim3(MAX_THREADS, 1, 1);
    initialize<<<grid, block>>>(d_out, 0);
    /**
     * Copy the host data to machine.
     */
    u8 *d_in;
    cudaMalloc((void **) &d_in, length * sizeof(u8));
    gpuErrchk(cudaMemcpy(d_in, h_in, length * sizeof(u8), cudaMemcpyHostToDevice));
    /**
     * Call kernel m_hist() to count the individual values.
     */
    grid = dim3((int) ceil((float) length/MAX_THREADS));
    //std::cout<<sizeof(bool);
    
    cudaFuncSetCacheConfig(m_hist, cudaFuncCachePreferShared);
    m_hist<<<grid, block>>>(d_in, d_out, length);
    cudaMemcpy(h_out, d_out, MAX_BINS * sizeof(u32), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    //std::cout<<"Time taken "<<time<<std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
