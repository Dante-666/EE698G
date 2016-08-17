#ifndef __PARALLEL_H__
#define __PARALLEL_H__

#include <math.h>
#include <cuda_runtime.h>

#define f32 float
#define u32 unsigned int
#define u8  unsigned char
#define MAX_BINS 256

#define MAX_THREADS 1024
#define MAX_THREADX 32
#define MAX_THREADY 32

/**
 * Helper kernel to initialize the device array to 0 count.
 */
__global__ void init(u32 *d_out, u32 value);

/**
 * Helper kernel to calculate the marginal histogram.
 */
__global__ void m_hist(u8 *d_in, u32 *d_out, u32 length);

/**
 * Helper kernel to calculate the joint histogram.
 */

/**
 * This function returns the marginal histogram from a sample stream.
 * Returns 0 on success and -1 upon failure.
 */
extern int m_histogram(u8 *h_in, u32 *h_out, u32 length);

/**
 * This function returns the joint histogram for two images.
 * Returns 0 in success and -1 upon failure
 */
extern int j_histogram();

#endif
