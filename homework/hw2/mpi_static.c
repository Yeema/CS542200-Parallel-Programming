#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#define MAX_ITER 10000

void write_png(const char* filename, const int width, const int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            // row[x * 3] = ((p & 0xf) << 4);
            png_bytep color = row + x * 3;
            if (p != MAX_ITER) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* argument parsing */
    assert(argc == 9);   
    int num_threads = strtol(argv[1], 0, 10);
    double left = strtod(argv[2], 0);
    double right = strtod(argv[3], 0);
    double lower = strtod(argv[4], 0);
    double upper = strtod(argv[5], 0);
    int width = strtol(argv[6], 0, 10);
    int height = strtol(argv[7], 0, 10);
    const char* filename = argv[8];

    /* allocate memory for image */
    int size_image = width * height;

    /**************** MPI Init ****************/
    MPI_Init(&argc, &argv);
    // double time_start = MPI_Wtime();
    int rank, nums_Proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nums_Proc);

    int* image;
    int* sum_image;
    // image = (int*)malloc(size_image * sizeof(int));
    // memset(image, 0, size_image * sizeof(int));
    image = (int *)calloc(size_image, sizeof(int)); 
    if (rank == nums_Proc-1 && rank!=0){
        // sum_image = (int*)malloc(size_image * sizeof(int));
        // memset(sum_image, 0, size_image * sizeof(int));
        sum_image = (int *)calloc(size_image, sizeof(int)); 
    }
    /**************** setting up start and end ****************/
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        int start = rank - ((j*width) % nums_Proc) < 0 ? rank + nums_Proc - ((j*width) % nums_Proc): rank - ((j*width) % nums_Proc);
        for (int i = start ; i < width; i+=nums_Proc) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < MAX_ITER && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }
    /**************** message passing ****************/

    if(nums_Proc-1 > 0){
        MPI_Reduce(image , sum_image, size_image, MPI_INT, MPI_SUM , nums_Proc - 1 ,MPI_COMM_WORLD);
        if(nums_Proc-1==rank){
            write_png(filename, width, height, sum_image);
            free(sum_image);
        }
    }else{
        write_png(filename, width, height, image);
    }
 
    free(image);

    // double time_end = MPI_Wtime();
    // printf("Rank %d, Time: %f sec\n", rank, time_end - time_start);

    MPI_Finalize();
}