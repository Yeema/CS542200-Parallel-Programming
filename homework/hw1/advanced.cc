#include <iostream> 
#include <algorithm> 
#include <vector>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atoi */
#include <mpi.h>
#include <string.h>
using namespace std; 
void timing(double *time,int rank){
    if (rank==0)
        *time = (double)clock();
}
void aggrTiming(double *sumup , double head , double tail,int rank){
    if(rank==0){
        *sumup += tail - head;
    }
}
int sorted = 0;
bool iseven(int a){
    return a%2==0? true:false;
}
bool isodd(int a){
    return a%2==0? false:true;
}
void mymemcpy(void * dest, void * src, int n)
{
    int i ;
    float *Dest = (float *)dest ;
    float *Src = (float *)src ;
    for(i=0;i<(n>>2);i++)         
            Dest[i] = Src[i] ;
}
void mergeArrays(float arr1[], float arr2[], int n1, int n2, float arr3[]) 
{ 
    int i = 0, j = 0, k = 0; 
  
    // Traverse both array 
    while (i<n1 && j <n2) 
    { 
        // Check if current element of first 
        // array is smaller than current element 
        // of second array. If yes, store first 
        // array element and increment first array 
        // index. Otherwise do same with second array 
        if (arr1[i] <= arr2[j]) 
            arr3[k++] = arr1[i++]; 
        else{
            arr3[k++] = arr2[j++];
            sorted = 0;
        } 
    } 
    // // Store remaining elements of first array 
    while (i < n1) 
        arr3[k++] = arr1[i++]; 
  
    // Store remaining elements of second array 
    while (j < n2) 
        arr3[k++] = arr2[j++]; 
    // if(i<n1)
    //     mymemcpy(arr3+k, arr1+i, (n1-i)*sizeof(float));
    // else
    //     mymemcpy(arr3+k, arr2+j, (n2-j)*sizeof(float));
} 
int main(int argc, char** argv) 
{   double start = (double)clock(), end,latency=0, startIo , endIo,io = 0, startComm, endComm, comm = 0;
    int numtasks,rank;
    int count;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num = atoi(argv[1]);
    if (num < numtasks)
        numtasks = num;
    int rankSize = (num + numtasks - 1)/numtasks; // (5-1)/4 + 1 or (4-1)/4 + 1
    timing(&startIo,rank);
    //open file
    MPI_File fh;
    MPI_Status status;
    //開檔案
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    //設定每個rank的offset
	MPI_File_set_view(fh, sizeof(float) * rankSize * rank, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    //讀取內容
    float *nums = new float[rankSize];
	MPI_File_read(fh, nums, rankSize, MPI_FLOAT, &status);  
    //每個rank實際讀出的數字量
	MPI_Get_count(&status, MPI_FLOAT, &count);
    int count_size = count * sizeof(float);
	MPI_File_close(&fh);
    timing(&endIo,rank);
    aggrTiming(&io,startIo,endIo,rank);
    float *combined = new float [2*rankSize];
    float *c_buffer = new float [rankSize];
    int count_buf = rank+1 < num/rankSize? rankSize : num%rankSize; 
    sort(nums,nums+count);
    if(numtasks==1){
        sorted = 1;
    }
    while(!sorted){
        sorted = 1;
            //odd P-->P-1
            //P-1 compare P-1-->P
            if(isodd(rank) && rank < numtasks){
                timing(&startComm,rank);
                MPI_Send(nums , rankSize , MPI_FLOAT , rank-1, 1 , MPI_COMM_WORLD);
                MPI_Recv(nums , rankSize , MPI_FLOAT , rank-1 , 0 , MPI_COMM_WORLD , &status);
                timing(&endComm,rank);
                aggrTiming(&comm, startComm , endComm, rank);
            }else if(iseven(rank)){
                if(rank+1 < numtasks){
                    timing(&startComm,rank);
                    MPI_Recv(c_buffer , rankSize , MPI_FLOAT , rank+1 , 1 , MPI_COMM_WORLD,&status);
                    timing(&endComm,rank);
                    aggrTiming(&comm, startComm , endComm, rank);
                    mergeArrays(nums,c_buffer,count,count_buf,combined);  
                    if (!sorted)              
                        mymemcpy(nums, combined , count_size);
                    timing(&startComm,rank);
                    MPI_Send( combined+count , rankSize , MPI_FLOAT , rank+1 , 0 , MPI_COMM_WORLD);
                    timing(&endComm,rank);
                    aggrTiming(&comm, startComm , endComm, rank);
                }
            }
            // MPI_Barrier(MPI_COMM_WORLD);
            // even Q --> Q-1
            // Q-1 compare Q-1-->Q
            if(iseven(rank) && rank < numtasks){
                if(rank>0){
                    timing(&startComm,rank);
                    MPI_Send(nums , rankSize , MPI_FLOAT , rank-1 , 1 , MPI_COMM_WORLD);
                    MPI_Recv(nums , rankSize , MPI_FLOAT , rank-1 , 0 , MPI_COMM_WORLD,&status);
                    timing(&endComm,rank);
                    aggrTiming(&comm, startComm , endComm, rank);
                }
            }else if(isodd(rank)){
                if(rank+1<numtasks){
                    timing(&startComm,rank);
                    MPI_Recv(c_buffer,rankSize,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD,&status);
                    timing(&endComm,rank);
                    aggrTiming(&comm, startComm , endComm, rank);
    
                    mergeArrays(nums, c_buffer , count, count_buf,combined);
                    if (!sorted)
                        mymemcpy(nums, combined , count_size);

                    timing(&startComm,rank);
                    MPI_Send( combined+count , rankSize , MPI_FLOAT , rank+1 , 0 , MPI_COMM_WORLD);
                    timing(&endComm,rank);
                    aggrTiming(&comm, startComm , endComm, rank);
                }
            }
            int tmp = sorted;
            timing(&startComm,rank);
            MPI_Allreduce(&tmp, &sorted, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);
            timing(&endComm,rank);
            aggrTiming(&comm, startComm , endComm, rank);
    } 
    timing(&startIo,rank);
    // write file
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh); 
	MPI_File_set_view(fh, sizeof(float) * rankSize * rank,  MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
	MPI_File_write(fh, nums, count, MPI_FLOAT, &status);
	MPI_File_close(&fh);
    timing(&endIo,rank);
    aggrTiming(&io,startIo,endIo,rank);
    delete []nums;
    delete []combined;
    delete []c_buffer;
    timing(&end, rank);
    aggrTiming(&latency, start,end,rank);
    if(rank==0)
        printf("CPU : %lf\nIO : %lf\nCommunication : %lf\nTotal : %lf\n" , (latency-io-comm)/CLOCKS_PER_SEC , io/CLOCKS_PER_SEC , comm/CLOCKS_PER_SEC , latency/CLOCKS_PER_SEC);
    MPI_Finalize();
} 