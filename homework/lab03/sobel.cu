#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE  8
unsigned char *device_s = NULL;
unsigned char *device_t = NULL;
unsigned char *host_s = NULL;       // source image array
unsigned char *host_t = NULL;       // target image array
FILE *fp_s = NULL;                  // source file handler
FILE *fp_t = NULL;                  // target file handler

unsigned int   width, height;       // image width, image height
unsigned int   rgb_raw_data_offset; // RGB raw data offset
unsigned char  bit_per_pixel;       // bit per pixel
unsigned short byte_per_pixel;      // byte per pixel

// bitmap header
unsigned char header[54] = {
	0x42,        // identity : B
	0x4d,        // identity : M
	0, 0, 0, 0,  // file size
	0, 0,        // reserved1
	0, 0,        // reserved2
	54, 0, 0, 0, // RGB data offset
	40, 0, 0, 0, // struct BITMAPINFOHEADER size
	0, 0, 0, 0,  // bmp width
	0, 0, 0, 0,  // bmp height
	1, 0,        // planes
	24, 0,       // bit per pixel
	0, 0, 0, 0,  // compression
	0, 0, 0, 0,  // data size
	0, 0, 0, 0,  // h resolution
	0, 0, 0, 0,  // v resolution 
	0, 0, 0, 0,  // used colors
	0, 0, 0, 0   // important colors
};

// sobel mask (5x5 version)
// Task 2: Put mask[][][] into Shared Memroy
__device__ int dev_mask[MASK_N][MASK_X][MASK_Y] = {
	{{ -1, -4, -6, -4, -1},
	 { -2, -8,-12, -8, -2},
	 {  0,  0,  0,  0,  0},
	 {  2,  8, 12,  8,  2},
	 {  1,  4,  6,  4,  1}},
	{{ -1, -2,  0,  2,  1},
	 { -4, -8,  0,  8,  4},
	 { -6,-12,  0, 12,  6},
	 { -4, -8,  0,  8,  4},
	 { -1, -2,  0,  2,  1}}
};

int read_bmp (const char *fname_s) {
	fp_s = fopen(fname_s, "rb");
	if (fp_s == NULL) {
		printf("fopen fp_s error\n");
		return -1;
	}

	// move offset to 10 to find rgb raw data offset
	fseek(fp_s, 10, SEEK_SET);
	fread(&rgb_raw_data_offset, sizeof(unsigned int), 1, fp_s);

	// move offset to 18 to get width & height;
	fseek(fp_s, 18, SEEK_SET); 
	fread(&width,  sizeof(unsigned int), 1, fp_s);
	fread(&height, sizeof(unsigned int), 1, fp_s);

	// get bit per pixel
	fseek(fp_s, 28, SEEK_SET); 
	fread(&bit_per_pixel, sizeof(unsigned short), 1, fp_s);
	byte_per_pixel = bit_per_pixel / 8;

	// move offset to rgb_raw_data_offset to get RGB raw data
	fseek(fp_s, rgb_raw_data_offset, SEEK_SET);

	// Task 3: Assign host_s to "Pinnned Memory" in CPU malloc doesn't pin
	// Hint  : err = cudaMallocHost ( ... )
	//         if (err != CUDA_SUCCESS)
	// host_s = (unsigned char *) malloc((size_t)width * height * byte_per_pixel);
	// if (host_s == NULL) {
	// 	printf("malloc images_s error\n");
	// 	return -1;
	// }
	cudaMallocHost(&host_s, (size_t)width * height * byte_per_pixel);

	// Task 3: Assign host_t to Pinned Memory
	// Hint  : err = cudaMallocHost ( ... )
	//         if (err != CUDA_SUCCESS)
	// host_t = (unsigned char *) malloc((size_t) width * height * byte_per_pixel);
	// if (host_t == NULL) {
	// 	printf("malloc host_t error\n");
	// 	return -1;
	// }
	cudaMallocHost(&host_t, (size_t)width * height * byte_per_pixel);

	fread(host_s, sizeof(unsigned char), (size_t)(long) width * height * byte_per_pixel, fp_s);

	return 0;
}

// Executed on the device Callable from the host only
__global__  void sobel (const int width, const int height, const int byte_per_pixel, unsigned char* host_s, unsigned char* host_t) {
	int  x, y, i, v, u;            // for loop counter
	int  R, G, B;                  // color of R, G, B
	double val[MASK_N*3] = {0.0};
	int adjustX, adjustY, xBound, yBound;
	
	// Task 2: Put mask[][][] into Shared Memory
	// Hint  : Please declare it in kernel function
	//         Then use some threads to move data from global memory to shared memory
	//         Remember to __syncthreads() after it's done <WHY?>

	__shared__ int mask[MASK_N][MASK_X][MASK_Y];
	// for(int i = blockIdx.x ; i  <MASK_N; i+=gridDim.x) {
	// 	for(int j = 0 ; j< MASK_X ; j+=)
	// 		for(int k = 0;k<MASK_Y;k++)
	// 			mask[i][j][k] = dev_mask[i][j][k];
	// }
	x = (blockIdx.x * blockDim.x + threadIdx.x)%MASK_N;
	y = (blockIdx.x * blockDim.x + threadIdx.x)/MASK_N%MASK_X;
	i = (blockIdx.x * blockDim.x + threadIdx.x)/MASK_N/MASK_X%MASK_Y;
	mask[x][y][i] = dev_mask[x][y][i];
	__syncthreads();// wait for each thread to copy its elemenet

	if ((blockIdx.x * blockDim.x + threadIdx.x) == height)
	{
		for(x =0;x < MASK_N ; x++)
		{
			for(y=0;y<MASK_X;y++)
			{
				for(i=0;i<MASK_Y;i++)
					printf("%d ",mask[x][y][i]);
				printf("\n");
			}
			
		}	
	}
	// Task 1: Relabel x or y or both into combination of blockIdx, threadIdx ... etc
	// Hint A: We do not have enough threads for each pixels in the image, so what should we do?
	// Hint B: Maybe you can map each y to different threads in different blocks
	y = blockIdx.x * blockDim.x + threadIdx.x;
	// for (y = 0; y < height; ++y) {
	if(y<height){
		for (x = 0; x < width; ++x) {
			for (i = 0; i < MASK_N; ++i) {
				adjustX = (MASK_X % 2) ? 1 : 0;
				adjustY = (MASK_Y % 2) ? 1 : 0;
				xBound = MASK_X /2;
				yBound = MASK_Y /2;

				val[i*3+2] = 0.0;
				val[i*3+1] = 0.0;
				val[i*3] = 0.0;

				for (v = -yBound; v < yBound + adjustY; ++v) {
					for (u = -xBound; u < xBound + adjustX; ++u) {
						if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
							R = host_s[byte_per_pixel * (width * (y+v) + (x+u)) + 2];
							G = host_s[byte_per_pixel * (width * (y+v) + (x+u)) + 1];
							B = host_s[byte_per_pixel * (width * (y+v) + (x+u)) + 0];
							val[i*3+2] += R * mask[i][u + xBound][v + yBound];
							val[i*3+1] += G * mask[i][u + xBound][v + yBound];
							val[i*3+0] += B * mask[i][u + xBound][v + yBound];
						}	
					}
				}
			}

			double totalR = 0.0;
			double totalG = 0.0;
			double totalB = 0.0;
			for (i = 0; i < MASK_N; ++i) {
				totalR += val[i * 3 + 2] * val[i * 3 + 2];
				totalG += val[i * 3 + 1] * val[i * 3 + 1];
				totalB += val[i * 3 + 0] * val[i * 3 + 0];
			}

			totalR = sqrt(totalR) / SCALE;
			totalG = sqrt(totalG) / SCALE;
			totalB = sqrt(totalB) / SCALE;
			const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
			const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
			const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
			host_t[byte_per_pixel * (width * y + x) + 2] = cR;
			host_t[byte_per_pixel * (width * y + x) + 1] = cG;
			host_t[byte_per_pixel * (width * y + x) + 0] = cB;
		}
	}else{
		return;
	}
}

int write_bmp (const char *fname_t) {
	unsigned int file_size;

	fp_t = fopen(fname_t, "wb");
	if (fp_t == NULL) {
		printf("fopen fname_t error\n");
		return -1;
	}

	// file size  
	file_size = width * height * byte_per_pixel + rgb_raw_data_offset;
	header[2] = (unsigned char)(file_size & 0x000000ff);
	header[3] = (file_size >> 8)  & 0x000000ff;
	header[4] = (file_size >> 16) & 0x000000ff;
	header[5] = (file_size >> 24) & 0x000000ff;

	// width
	header[18] = width & 0x000000ff;
	header[19] = (width >> 8)  & 0x000000ff;
	header[20] = (width >> 16) & 0x000000ff;
	header[21] = (width >> 24) & 0x000000ff;

	// height
	header[22] = height &0x000000ff;
	header[23] = (height >> 8)  & 0x000000ff;
	header[24] = (height >> 16) & 0x000000ff;
	header[25] = (height >> 24) & 0x000000ff;

	// bit per pixel
	header[28] = bit_per_pixel;

	// write header
	fwrite(header, sizeof(unsigned char), rgb_raw_data_offset, fp_t);

	// write image
	fwrite(host_t, sizeof(unsigned char), (size_t)(long)width * height * byte_per_pixel, fp_t);

	fclose(fp_s);
	fclose(fp_t);

	return 0;
}

int main(int argc, char **argv) {

    assert(argc == 3);
    const char* input = argv[1];
    const char* output = argv[2];
	read_bmp(input); // 24 bit gray level image
    

	// Task 1: Allocate memory on GPU
	// Hint  : cudaMalloc ()
	//         What do we need to store on GPU? (input image, output image, ...)

	// Task 1: Memory copy from Host to Device (GPU)
	// Hint  : cudaMemcpy ( ... , cudaMemcpyHostToDevice )

	// Task 1: Modify sobel() to CUDA kernel function
	// Hint  : sobel_Kernel <<< ??? , ??? >>> ( ??? );
	// sobel();
	cudaMalloc((void**)&device_s, (size_t) width * height * byte_per_pixel);
    cudaMalloc((void**)&device_t, (size_t) width * height * byte_per_pixel);
    cudaMemcpy(device_s, host_s, (size_t) width * height * byte_per_pixel, cudaMemcpyHostToDevice);
	sobel<<<(height/256) + 1, 256>>>(width, height, byte_per_pixel, device_s, device_t);
	// Task 1: Memory Copy from Device (GPU) to Host
	// Hint  : cudaMemcpy ( ... , cudaMemcpyDeviceToHost )
	cudaMemcpy(host_t, device_t, (size_t) width * height * byte_per_pixel, cudaMemcpyDeviceToHost);
	// Task 1: Free memory on device
	// Hint  : cudaFree ( ... )
	cudaFree(device_s);
    cudaFree(device_t);
	write_bmp(output);

	// Task 3: Free Pinned memory
	// Hint  : replace free ( ... ) by cudaFreeHost ( ... )
	// free (host_s);
	// free (host_t);
	cudaFreeHost(host_s);
    cudaFreeHost(host_t);
}
