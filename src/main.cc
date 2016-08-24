#include "parallel.h"
#include "config.h"
#include "data.h"

int init(const char* filename){

    if (!filename) return -1;
    
    FILE *cf;
    if (!(cf = fopen(filename, "r"))) return -1;

    Config *configHandle;
    if(!(configHandle = config_parse_file(cf, NULL))) return -1;

    char *scanFolder;
    config_get_str(configHandle, "calibration.scan.scan_folder", &scanFolder);
    char *scanBaseName;
    config_get_str(configHandle, "calibration.scan.scan_base_name", &scanBaseName);
    char *scanType;
    config_get_str(configHandle, "calibration.scan.scan_type", &scanType);
    int numScans;
    config_get_int(configHandle, "calibration.scan.num_scans_used", &numScans);
    int scansRandom;
    config_get_int(configHandle, "calibration.scan.scans_randomly_sampled", &scansRandom);
    int totalScans;
    config_get_int(configHandle, "calibration.scan.total_scans", &totalScans);
    int *useScans = (int *) malloc (sizeof(int) * numScans);
    if (scansRandom)
        config_get_random_numbers(1, totalScans, useScans, numScans);
    else
        config_get_int_array(configHandle, "calibration.scan.use_scans", useScans, numScans);

    int numCams;
    config_get_int(configHandle, "calibration.cameras.num_cameras", &numCams);
    char *camFolder;
    char *imageBaseName;
    config_get_str(configHandle, "calibration.cameras.image_base_name", &imageBaseName);
    char *imageType;
     config_get_str(configHandle, "calibration.cameras.image_type", &imageType);

    for (int s = 0; s < numScans; s++) {
        char scanFile[256];
        sprintf(scanFile, "%s%s%04d.%s", scanFolder, scanBaseName, useScans[s], scanType);
        printf("%s\n", scanFile);

        loadLidarData(scanFile);
        for (int i = 0; i < numCams; i++) {
            char camFile[256];
            char str[256];
            sprintf(str, "calibration.cameras.camera_%d.folder", i);
            config_get_str(configHandle, str, &camFolder);

            sprintf(camFile, "%s%s%04d.%s", camFolder, imageBaseName, useScans[s], imageType);
            printf("%s\n", camFile);
//            loadCamData(camFile);
        }
    }

    for (int i = 0; i < numCams; i++) {
        char *maskFile;
        char str[256];
        sprintf(str, "calibration.cameras.camera_%d.mask", i);
        config_get_str(configHandle, str, &maskFile);
        printf("%s\n", maskFile);
        //loadMask(maskFile);
    }

    free(imageType);
    free(imageBaseName);
    free(scanFolder);
    free(scanBaseName);
    free(scanType);
    free(useScans);

    return 0;
}

int main() {
    if (init(DEFAULT_CONFIG_PATH) < 0) {
        fprintf(stderr, "Initialization failed, check the config file...");
    }
    /*if (argc < 2 || argc > 2) return -1;
    FreeImage_Initialise(); 
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(argv[1]);
    FIBITMAP* immap = FreeImage_Load(format, argv[1]);

    FIBITMAP* greymap = FreeImage_ConvertToGreyscale(immap); 
    FREE_IMAGE_TYPE type = FreeImage_GetImageType(greymap);

    int numRows = FreeImage_GetHeight(greymap);
    int numCols = FreeImage_GetWidth(greymap);

    u8 *h_in = (u8 *) malloc(numRows * numCols * sizeof(u8));
    u32 *h_out = (u32 *) malloc(MAX_BINS * sizeof(u32)); 
 
    int i = 0;
    if(type == FIT_BITMAP) {
        BYTE *bits = (BYTE *) FreeImage_GetBits(greymap);
        for(int y = 0; y < numRows * numCols; y++) {
            h_in[i++] = (u8) *bits;
            bits += 1;
        }
    }

    FreeImage_Save(FIF_PNG, greymap, "grey.png", PNG_DEFAULT);
    FreeImage_DeInitialise();

    //auto start = std::chrono::high_resolution_clock::now();

    m_histogram(h_in, h_out, numRows * numCols);

    //auto stop = std::chrono::high_resolution_clock::now() - start;
    //std::cout<<"Time taken "<<std::chrono::duration_cast<std::chrono::microseconds>(stop).count()<<std::endl;

    FILE *f = fopen("hist.txt", "w");

    for (i = 0; i < MAX_BINS; i++) fprintf(f, "%d ", h_out[i]);

    fclose(f);

    free(h_in);
    free(h_out);*/

    return 0;
}
