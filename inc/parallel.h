#ifndef __PARALLEL_H__
#define __PARALLEL_H__

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define f32 float
#define u32 unsigned int
#define u8  unsigned char
#define MAX_BINS 256

#define MAX_THREADS 1024
#define MAX_THREADX 32
#define MAX_THREADY 32

typedef struct _Point3D Point3D_t;
struct _Point3D {
    f32 x;
    f32 y;
    f32 z;
    u8 refc;

    f32 range;
};

typedef struct _PointCloud PointCloud_t;
struct _PointCloud {
    thrust::device_vector<Point3D_t> points;
};

typedef struct _PointCloudArray PointCloudArray_t;
struct _PointCloudArray {
    thrust::device_vector<PointCloud_t> array;
};

typedef struct _RGB RGB_t;
struct _RGB {
    u8 R;
    u8 G;
    u8 B;
};

typedef struct _RawImage RawImage_t;
struct _RawImage {
    thrust::device_vector<RGB_t> rawImage;
};

typedef struct _ImageArray ImageArray_t;
struct _ImageArray {
    thrust::device_vector<RawImage_t> imageArray;
};

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
int m_histogram(u8 *h_in, u32 *h_out, u32 length);
int histogram(u8 *h_in, u32 *h_out, u32 length);

/**
 * This function returns the joint histogram for two images.
 * Returns 0 in success and -1 upon failure
 */
//extern int j_histogram();

#endif
