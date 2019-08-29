#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define channel_master_send 0
#define channel_slave_send 1
#define data_tag 0
#define terminate_tag 1
#define master 0
#define MAX_ITER 10000
void ms_singleProc(int height, int width, double upper, double lower, double right, double left, int* image){
    /* mandelbrot set */
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < 100000 && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }
}

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
    MPI_Request request;
    /**************** MPI Init ****************/
    MPI_Init(&argc, &argv);
    // double time_start = MPI_Wtime();
    int rank, nums_Proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nums_Proc);

    int singleProc = nums_Proc == 1 ? 1 : 0;
    int size_image = width * height;
    int* image;
    int* sum_image;
    image = (int *)calloc(size_image, sizeof(int)); 
    if (rank ==0){
        sum_image = (int *)calloc(size_image, sizeof(int)); 
    }
    if(!singleProc){        
        int master_send[2]; // [row, data_tag]
        int slave_send[2]; // [rank_salve, row_complete]

        // master
        if(rank == 0){
            master_send[0] = 0;
            master_send[1] = data_tag;
            int count = 0;

            for(int k=1; k<nums_Proc; k++){
                MPI_Isend(master_send, 2, MPI_INT, k, channel_master_send, MPI_COMM_WORLD,&request);
                count++;
                master_send[0]++;
            }

            do{
                MPI_Recv(slave_send, 2, MPI_INT, MPI_ANY_SOURCE, channel_slave_send, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                count--;
                if(master_send[0] < height){
                    MPI_Isend(master_send, 2, MPI_INT, slave_send[0], channel_master_send, MPI_COMM_WORLD,&request);
                    count++;
                    master_send[0]++;
                }else{
                    master_send[1] = terminate_tag;
                    MPI_Isend(master_send, 2, MPI_INT, slave_send[0], channel_master_send, MPI_COMM_WORLD,&request);
                }
            }while(count > 0);
            MPI_Reduce(sum_image , image, size_image, MPI_INT, MPI_SUM , 0 ,MPI_COMM_WORLD);
        }else{            
            //slave
            slave_send[0] = rank;
            MPI_Recv(master_send, 2, MPI_INT, master, channel_master_send, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            while(master_send[1] == data_tag){
                slave_send[1] = master_send[0];

                double y0 = slave_send[1] * ((upper - lower) / height) + lower;
                for (int i = 0 ; i < width; i++) {
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
                    image[slave_send[1] * width + i] = repeats;
                }

                MPI_Isend(slave_send, 2, MPI_INT, master, channel_slave_send, MPI_COMM_WORLD,&request);
                MPI_Recv(master_send, 2, MPI_INT, master, channel_master_send, MPI_COMM_WORLD, MPI_STATUS_IGNORE);           
            }
            MPI_Reduce(image , sum_image, size_image, MPI_INT, MPI_SUM , 0 ,MPI_COMM_WORLD);
        }

    }else{
        //do singleProc ver.
        ms_singleProc(height, width, upper, lower, right, left, image);
    }
    /* draw and cleanup */
    if(rank == 0){
        write_png(filename, width, height, image);
        free(sum_image);
    }
    free(image);
    // double time_end = MPI_Wtime();
    // printf("Rank %d, Time: %f sec\n", rank, time_end - time_start);

    MPI_Finalize();
}