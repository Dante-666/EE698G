#include <netpbm/pam.h>

#include "parallel.h"

/** These are the data which is absolutely needed by the host to pass
 * on to the device. Afterwards, we can do everything in the device 
 * memory.
 */
PointCloudArray_t d_lidarData;
ImageArray_t d_camData;
ImageArray_t d_mask;


/** Self-evident names.
 * I've avoided classes and used functions.
 */
int loadLidarData(const char *filename);

int loadCamData(const char *filename);

int loadMask(const char *filename);
