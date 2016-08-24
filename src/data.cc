#include "data.h"

int loadLidarData (const char *filename) {

    FILE *f;
    if (!(f = fopen(filename, "r"))) return -1;

    int numPoints;
    fscanf(f, "%d\n", &numPoints);

    double DIST_THRESH = 10000;

    PointCloud_t pCloud;
    while (!feof (f)) {
        Point3D_t point;
        fscanf (f, "%f %f %f %hhu\n", &point.x, &point.y, &point.z, &point.refc);
        double dist = point.x*point.x + point.y*point.y + point.z*point.z;
        point.range = dist/DIST_THRESH; 
        pCloud.points.push_back (point);
        numPoints++;
    }

    d_lidarData.array.push_back (pCloud);
    std::cout << "Num points loaded = " << pCloud.points.size () << std::endl;
    fflush (f);
    fclose (f);

    return 0;
}

int loadCamData(const char* filename) {
    struct pam image;
    FILE *f;
    if (!(f = fopen(filename, "r"))) return -1;
    
    tuple **stream;
    stream = pnm_readpam(f, &image, sizeof(struct pam));

        

    fflush (f);
    fclose (f);

    return 0;
}
