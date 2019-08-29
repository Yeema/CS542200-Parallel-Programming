#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <iostream> 
using namespace std;
const int INF = 1000000000;
const int V = 20000;
int n; 
unsigned int m;
static int Dist[V * V];
int* result;
__global__ void phase_one(int r, int n, int B, int* Dist)
{
    extern __shared__ int shared_Dist[];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int pivot_i = r*B + y;
    int pivot_j = r*B + x;
    
    // copy to shared memory
    shared_Dist[y*B + x] = (pivot_i<n && pivot_j<n) ? Dist[pivot_i*n + pivot_j] : INF;
    __syncthreads();

    //floyd-algo
    #pragma unroll
    for(int k=0; k<B; ++k){
        if(shared_Dist[y * B + x] > shared_Dist[y * B + k] + shared_Dist[k * B + x]){
            shared_Dist[y * B + x] = shared_Dist[y * B + k] + shared_Dist[k * B + x];
        }
        __syncthreads();
    }

    // update global memory
    if(pivot_i<n && pivot_j<n){
        Dist[pivot_i * n + pivot_j] = shared_Dist[y * B + x];
    }
}

__global__ void phase_two(int r, int n, int B, int* Dist)
{
    // pivot
    if(blockIdx.x == r) return;

    extern __shared__ int shared_mem[];
    
    int* shared_pivot = shared_mem;
    int* shared_Dist = shared_mem + B * B;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int pivot_i = r*B + y;
    int pivot_j = r*B + x;

    // pivot copy to shared memory
    shared_pivot[y*B + x] = (pivot_i < n && pivot_j < n)? Dist[pivot_i*n + pivot_j] : INF;
    __syncthreads();

    int block_i, block_j;
    // same row
    if(blockIdx.y == 0){                    
        block_i = pivot_i;
        block_j = blockIdx.x*B + x; 
    }else{   
        //same col                             
        block_i = blockIdx.x*B + y;
        block_j = pivot_j;
    }
    if(block_i >= n || block_j >= n) return;

    shared_Dist[y*B + x] = (block_i<n && block_j<n)? Dist[block_i*n + block_j] : INF;
    __syncthreads();

    // same row
    if(blockIdx.y == 0){        
        #pragma unroll
        for(int k=0; k<B; ++k){
            if(shared_Dist[y*B + x] > shared_pivot[y*B + k] + shared_Dist[k*B + x]){
                shared_Dist[y*B + x] = shared_pivot[y*B + k] + shared_Dist[k*B + x];
            }  
            __syncthreads();
        }
    }else{       
        // same col                
        #pragma unroll
        for(int k=0; k<B; ++k){
            if(shared_Dist[y*B + x] > shared_Dist[y*B + k] + shared_pivot[k*B + x]){
                shared_Dist[y*B + x] = shared_Dist[y*B + k] + shared_pivot[k*B + x];
            }        
            __syncthreads();
        }     
    }
    // update global
    if(block_i<n && block_j<n){
        Dist[block_i*n + block_j] = shared_Dist[y*B + x];
    }
}

__global__ void phase_three(int r, int n, int B, int* Dist)
{
    // pivot or same row or col
    if(blockIdx.x == r || blockIdx.y == r) return;  

    extern __shared__ int shared_mem[];
    int* shared_row = shared_mem;
    int* shared_col = shared_mem + B*B;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int block_i = blockIdx.y*B + y;
    int block_j = blockIdx.x*B + x;
    int row_i = r*B + y;
    int row_j = block_j;
    int col_i = block_i;
    int col_j = r*B + x;
    
    // copy same row,col with pivot to shared memory
    shared_row[y*B + x] = (row_i<n && row_j<n)? Dist[row_i*n + row_j] : INF;
    shared_col[y*B + x] = (col_i<n && col_j<n)? Dist[col_i*n + col_j] : INF;
    __syncthreads();

    if(block_i >= n || block_j >= n) return;

    // floyd-algo
    int target = Dist[block_i*n + block_j];      
    #pragma unroll
    for(int k=0; k<B; ++k){
        if(target > shared_col[y*B + k] + shared_row[k*B + x])
            target = shared_col[y*B + k] + shared_row[k*B + x];
    }
    // update global
    Dist[block_i*n + block_j] = target;
}

void input(char *inFileName)
{
    FILE* file = fopen(inFileName, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            if (i == j) {
                Dist[i*n + j] = 0;
            } else {
                Dist[i*n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++ i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName)
{
    ofstream ofile;
    ofile.open(outFileName, ios::binary | ios::out);
    for(int i=0; i<n ;i++){
        for(int j=0 ; j<n ; j++){
            ofile.write((char*) &result[i*n+j], sizeof(int));
        }
    }
    ofile.close();
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B)
{
	int num_blocks = ceil(n, B);						
	int* device_Dist;
	int shared_mem_size = sizeof(int)*n*n;

    // setup host to device
    cudaSetDevice(0);
	cudaMalloc(&device_Dist, shared_mem_size);
    cudaMemcpy(device_Dist, Dist, shared_mem_size, cudaMemcpyHostToDevice);
    result = (int*) malloc(shared_mem_size);
	
	dim3 block(B, B);
	dim3 grid_phase_one(1, 1), grid_phase_two(num_blocks, 2), grid_phase_three(num_blocks, num_blocks);			    
    
    // floyd-algo
	for(int i = 0; i < num_blocks; i++){
		phase_one<<< grid_phase_one, block, sizeof(int) * B*B >>>(i, n, B, device_Dist);
		phase_two<<< grid_phase_two, block, sizeof(int) * 2*B*B >>>(i, n, B, device_Dist);
		phase_three<<< grid_phase_three, block, sizeof(int) * 2*B*B >>>(i, n, B, device_Dist);
    }

    //device to host
	cudaMemcpy(result, device_Dist, shared_mem_size, cudaMemcpyDeviceToHost);
    cudaFree(device_Dist);
}



int main(int argc, char* argv[])
{
    input(argv[1]);
    
    // set block factor for experiment
    int B = argc>3? atoi(argv[3]) : n>32? 32 : n;
    block_FW(B);

    output(argv[2]);
	return 0;
}

