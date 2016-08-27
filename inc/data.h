#ifndef __DATA_H__
#define __DATA_H__

#include <netpbm/pam.h>

#include "parallel.h"
/**
 * Remove this after debugging.
 */
#include <iostream>

/** Self-evident names.
 * I've avoided classes and used functions.
 */
int loadLidarData(const char *filename, Point3D_t **data, int index);

int loadCamData(const char *filename, RGB_t **data, int index);

int loadMask(const char *filename, RGB_t **mask);

#endif
