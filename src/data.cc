#include "data.h"

int loadLidarData (const char *filename, Point3D_t **data, int index) {

    FILE *f;
    if (!(f = fopen(filename, "r"))) return -1;

    int numPoints;
    fscanf(f, "%d\n", &numPoints);

    double DIST_THRESH = 10000;

    //std::cout<<"here"<<std::endl;

    Point3D_t *cloud = (Point3D_t *) malloc(sizeof(Point3D_t) * numPoints);
    int i = 0;
    while (!feof (f)) {
        Point3D_t point;
        fscanf (f, "%f %f %f %hhu\n", &point.x, &point.y, &point.z, &point.refc);
        double dist = point.x*point.x + point.y*point.y + point.z*point.z;
        point.range = dist/DIST_THRESH; 
        cloud[i++] = point;
    }

    //std::cout<<"end"<<std::endl;

    data[index] = cloud;
    
    fflush (f);
    fclose (f);

    return 0;
}

int loadCamData(const char  *filename, RGB_t **data, int index) {
    struct pam *image;
    FILE *f;
    if (!(f = fopen(filename, "r"))) return -1;

    image = (struct pam *) malloc(sizeof(struct pam));
    
    //std::cout<<"here"<<std::endl;
    tuple **stream;
    stream = pnm_readpam(f, image, PAM_STRUCT_SIZE(tuple_type));

    //std::cout<<"here"<<std::endl;
    RGB_t *imstream = (RGB_t *) malloc(image->width * image->height * sizeof(RGB_t));
    for (int i = 0; i < image->height; i++) {
        for(int j = 0; j <  image->width; j++) {
        RGB_t pixel;
        tuple datum = stream[i][j];

        pixel.R = datum[0];
        pixel.G = datum[1];
        pixel.B = datum[2];

        imstream[i * image->width + j] = pixel;
        }
    }

    data[index] = imstream;

    //data[1] = &x[0];
    //std::cout<<"X = "<<x<<"is this"<<std::endl;
    //std::cout<<"end"<<std::endl;

    fflush (f);
    fclose (f);

    return 0;
}
