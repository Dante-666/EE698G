#include "data.h"

int loadLidarData (const char *filename, Point3D_t **data, int dataIndex) {

    FILE *f;
    if (!(f = fopen(filename, "r"))) return -1;

    int numPoints;
    fscanf(f, "%d\n", &numPoints);

    double DIST_THRESH = 10000;

    std::cout<<"here"<<std::endl;

    Point3D_t *cloud = (Point3D_t *) malloc(sizeof(Point3D_t) * numPoints);
    int i = 0;
    while (!feof (f)) {
        Point3D_t point;
        fscanf (f, "%f %f %f %hhu\n", &point.x, &point.y, &point.z, &point.refc);
        double dist = point.x*point.x + point.y*point.y + point.z*point.z;
        point.range = dist/DIST_THRESH; 
        cloud[i++] = point;
    }

    std::cout<<"end"<<std::endl;

    data[dataIndex] = cloud;
    
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
