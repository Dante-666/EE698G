#include <FreeImage.h>
#include <stdio.h>
#include "parallel.h"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 2) return -1;
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

    m_histogram(h_in, h_out, numRows * numCols);
    
    FILE *f = fopen("hist.txt", "w");

    for (i = 0; i < MAX_BINS; i++) fprintf(f, "%d ", h_out[i]);

    fclose(f);

    free(h_in);
    free(h_out);  

    return 0;
}
