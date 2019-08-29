#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream> 
using namespace std;
const int INF = 1000000000;
const int V = 20000;
int n; 
unsigned int m;	
static int Dist[V * V];
int num_procs;

__global__ void phase_one(int r, int n, int B, int* Dist)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int pivot_i = r*B + y;
    int pivot_j = r*B + x;
    extern __shared__ int shared_Dist[];
    // copy to shared memory
    shared_Dist[y*B + x] = (pivot_i<n && pivot_j<n)? Dist[pivot_i*n + pivot_j] : INF;
    __syncthreads();

    // floyd-algo
    #pragma unroll
    for(int k=0; k<B; ++k){
        if(shared_Dist[y*B + x] > shared_Dist[y*B + k] + shared_Dist[k*B + x]){
            shared_Dist[y*B + x] = shared_Dist[y*B + k] + shared_Dist[k*B + x];
        }
        __syncthreads();
    }
    // update global memory
    if(pivot_i<n && pivot_j<n){
        Dist[pivot_i*n + pivot_j] = shared_Dist[y*B + x];
    }
}

__global__ void phase_two(int r, int n, int B, int* Dist)
{
    // pivot
    if(blockIdx.x == r) return;            

    int x = threadIdx.x;
    int y = threadIdx.y;
    int pivot_i = r*B + y;
    int pivot_j = r*B + x;
    extern __shared__ int shared_mem[];
    int* shared_pivot = shared_mem;
    int* shared_Dist = shared_mem + B*B;
    
    // copy pivot to shared memory
    shared_pivot[y*B + x] = (pivot_i<n && pivot_j<n)? Dist[pivot_i*n + pivot_j] : INF;
    __syncthreads();

    int block_i, block_j;
    // same row
    if(blockIdx.y == 0){                    
        block_i = pivot_i;
        block_j = blockIdx.x*B + x; 
    }else{
        // same col                                 
        block_i = blockIdx.x*B + y;
        block_j = pivot_j;
    }

    if(block_i >= n || block_j >= n) return;
    // copy Dist to shared memory
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

    // copy to global memory
    if(block_i<n && block_j<n){
        Dist[block_i*n + block_j] = shared_Dist[y*B + x];
    }
}

__global__ void phase_three(int r, int n, int B, int* Dist, int bias)
{
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y + bias;
    // pivot or same row, col
    if(blockIdx_x == r || blockIdx_y == r) return;  

    extern __shared__ int shared_mem[];
    int* shared_row = shared_mem;
    int* shared_col = shared_mem + B*B;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int block_i = blockIdx_y*B + y;
    int block_j = blockIdx_x*B + x;
    int row_i = r * B + y;
    int row_j = block_j;
    int col_i = block_i;
    int col_j = r*B + x;
    
    // copy same row,col with pivot to shared memory
    shared_row[y*B + x] = (row_i<n && row_j<n)? Dist[row_i*n + row_j] : INF;
    shared_col[y*B + x] = (col_i<n && col_j<n)? Dist[col_i*n + col_j] : INF;
    __syncthreads();

    if(block_i >= n || block_j >= n) return;

    int target = Dist[block_i*n + block_j];     
    // floyd-algo
    #pragma unroll
    for(int k=0; k<B; ++k){
        if(target > shared_col[y*B + k] + shared_row[k*B + x]){
            target = shared_col[y*B + k] + shared_row[k*B + x];
        }
    }
    // update global memory
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
            ofile.write((char*) &Dist[i*n+j], sizeof(int));
        }
    }
    ofile.close();
}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B,int rank)
{
    // copy "Dist" at host memory to device
    int* device_Dist;
    int shared_mem_size = sizeof(int)*n*n;
    cudaSetDevice(rank);
    cudaMalloc(&device_Dist, shared_mem_size);    
    cudaMemcpy(device_Dist, Dist, shared_mem_size, cudaMemcpyHostToDevice);

    // speparate blocks
    int num_blocks = ceil(n, B);						         
    int avg_block = num_blocks / num_procs;
    // assign the rest to GPU0
    int start_round = (rank < num_blocks%num_procs)? (avg_block+1)*rank : avg_block*rank + num_blocks%num_procs;
    int block_row = (rank < num_blocks%num_procs)? avg_block+1 : avg_block;      
       

    dim3 block(B, B);
    dim3 grid_phase1(1, 1);                     
    dim3 grid_phase2(num_blocks, 2);			    	
    dim3 grid_phase3(num_blocks, block_row);	   

    // floyd-algo
    for(int r = 0; r < num_blocks; ++r){
        phase_one<<< grid_phase1, block, sizeof(int) * B*B >>>(r, n, B, device_Dist);
        phase_two<<< grid_phase2, block, sizeof(int) * 2*B*B >>>(r, n, B, device_Dist);
        phase_three<<< grid_phase3, block, sizeof(int) * 2*B*B >>>(r, n, B, device_Dist, start_round);
    
        // transfer pivot
        if(r < num_blocks - 1){
            int transfer_size = (n >= (r+2)*B)? sizeof(int)*B*n : sizeof(int)*n*(n - (r+1)*B);
            int transfer_row = r+1;
            // pivot at GPU0: r+1 < start_round + block_row
            // pivot at GPU1: r+1 ≥ start_round
            if(r+1 < start_round + block_row && r+1 >= start_round) {   
                // send pivot
                cudaMemcpy(Dist + transfer_row*n*B, device_Dist + transfer_row*n*B, transfer_size, cudaMemcpyDeviceToHost);
                for(int i = 0; i<num_procs; i++){
                    if(i != rank){                      
                        MPI_Send(Dist + transfer_row*n*B, transfer_size / sizeof(int), MPI_INT, i, 0, MPI_COMM_WORLD);                                   
                    }
                }               
            }else{       
                // recv pivot                
                MPI_Recv(Dist + transfer_row*n*B, transfer_size/sizeof(int), MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cudaMemcpy(device_Dist + transfer_row*n*B, Dist + transfer_row*n*B, transfer_size , cudaMemcpyHostToDevice);
            }
        }                      
    }
    
    int size = (n >= (start_round+block_row)*B)? sizeof(int)*B*n*block_row : sizeof(int)*n*(n - start_round*B); 
    // GPU0(upper) GPU1(lower) transfer to host
    // rank0 merge!
    if(rank == 0){
        cudaMemcpy(Dist + start_round*n*B, device_Dist + start_round*n*B, size, cudaMemcpyDeviceToHost);
        int buff[2];
        MPI_Recv(buff, 2, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);        
        MPI_Recv(Dist+buff[0], buff[1], MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }else{
        int* buff = (int*) malloc(size);
        cudaMemcpy(buff, device_Dist + start_round*n*B, size, cudaMemcpyDeviceToHost);
        // offset, size
        int shared_mem_sizes[2]={start_round*n*B , size/sizeof(int)};
        MPI_Send(shared_mem_sizes, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(buff, shared_mem_sizes[1], MPI_INT, 0, 0, MPI_COMM_WORLD);
    } 
    cudaFree(device_Dist);
}
int main(int argc, char* argv[])
{  
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0)
        input(argv[1]);
    // only rank0 has complete Dist,m,n
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Dist, n*n, MPI_INT, 0, MPI_COMM_WORLD);
    // set block factor for experiment
    int B = argc>3? atoi(argv[3]) : n>32? 32 : n;
    block_FW(B,rank);

    if(rank == 0){
        output(argv[2]);
    }

    MPI_Finalize();
	return 0;
}