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
    __shared__ u32 bin[MAX_BINS];

    if (threadIdx.x < MAX_BINS) bin[threadIdx.x] = 0;
    
    bin[d_in[id]]++;

    if (threadIdx.x < MAX_BINS - 1) atomicAdd(&d_out[threadIdx.x], bin[threadIdx.x]); 
}

int m_histogram(u8 *h_in, u32 *h_out, u32 length) {
    /**
     * Allocate memory for the bins [0-255]
     */
    u32 *d_out;
    cudaMalloc((void **) &d_out, MAX_BINS * sizeof(u32));
    /**
     * Call kernel initialize() to initialize all values to 0.
     */
    dim3 grid = dim3((int) ceil((float) MAX_BINS/MAX_THREADS));
    dim3 block = dim3(MAX_THREADS, 1, 1);
    printf("%d\n", MAX_BINS);
    printf("%d\n", MAX_THREADS);
    printf("%f\n",ceil((float) MAX_BINS/MAX_THREADS));
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
    grid = dim3(ceil(length/MAX_THREADS));
    m_hist<<<grid, block>>>(d_in, d_out, length);
    cudaMemcpy(h_out, d_out, MAX_BINS * sizeof(u32), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
