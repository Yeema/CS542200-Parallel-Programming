#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h>
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
int iseven(int a){
    return a%2==0? 1:0;
}
int isodd(int a){
    return a%2==0? 0:1;
}

void swap(float* a, float* b){
	float temp = *a; 
	*a = *b;
	*b = temp;
	return;
}

int main(int argc, char** argv)
{   double start = (double)clock(), end,latency=0, startIo , endIo,io = 0, startComm, endComm, comm = 0;
    int i,odd_even;
    int count;
    int numtasks,rank;
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
    float *nums = (float*) malloc(rankSize*sizeof(float));
	MPI_File_read(fh, nums, rankSize, MPI_FLOAT, &status);  
    //每個rank實際讀出的數字量
	MPI_Get_count(&status, MPI_FLOAT, &count);
	MPI_File_close(&fh);
    timing(&endIo,rank);
    aggrTiming(&io,startIo,endIo,rank);
    while (!sorted) {
        sorted = 1;
        //rank 內要有數字才需要排序
        if (numtasks > 1){
            //even
            //1 even communication sort
            if (rank<numtasks){
                float buffer;
                MPI_Status status;
                // inter change以even rank角度
                if(iseven(rank)){
                    if(rank+1<numtasks){
                        timing(&startComm,rank);
                        MPI_Send(&nums[count-1],1,MPI_FLOAT,rank+1,0,MPI_COMM_WORLD);
                        MPI_Recv(&buffer,1,MPI_FLOAT,rank+1,1,MPI_COMM_WORLD,&status);
                        timing(&endComm,rank);
                        aggrTiming(&comm, startComm , endComm, rank);
                        if(nums[count-1] > buffer){
                            nums[count-1] = buffer;
                            sorted = 0;
                        }
                    }
                }else{
                    // inter change以odd rank角度
                    //if rank-1<0 不會發生
                    timing(&startComm,rank);
                    MPI_Send(&nums[0],1,MPI_FLOAT,rank-1,1,MPI_COMM_WORLD);
                    MPI_Recv(&buffer,1,MPI_FLOAT,rank-1,0,MPI_COMM_WORLD,&status);
                    timing(&endComm,rank);
                    aggrTiming(&comm, startComm , endComm, rank);
                    if(nums[0] < buffer){
                        nums[0] = buffer;
                        sorted = 0;
                    }
                }
            }
            timing(&startComm,rank);
            MPI_Barrier(MPI_COMM_WORLD);
            timing(&endComm,rank);
            aggrTiming(&comm, startComm , endComm, rank);
            //2 even inner sort
            // innersort(nums, 0 , count);
            if(rank<numtasks){
                for (i = 0; i < count-1; i += 2){
                    if (nums[i] > nums[i+1]){ 
                        swap(&nums[i], &nums[i+1]);
                        sorted = 0;
                    }
                }
            }
            //odd
            //1 odd communication sort
            if(rank<numtasks){
                float buffer;
                MPI_Status status;
                if(isodd(rank)){
                    if(rank+1<numtasks){
                        timing(&startComm,rank);
                        MPI_Send(&nums[count-1], 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
                        MPI_Recv(&buffer, 1, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &status);
                        timing(&endComm,rank);
                        aggrTiming(&comm, startComm , endComm, rank);
                        if (nums[count-1] > buffer){
                            nums[count-1] = buffer;
                            sorted = 0;
                        }
                    }
                }
                else{
                    if(rank>0){
                        timing(&startComm,rank);
                        MPI_Send(&nums[0], 1, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD);
                        MPI_Recv(&buffer, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
                        timing(&endComm,rank);
                        aggrTiming(&comm, startComm , endComm, rank);
                        if(nums[0] < buffer){
                            nums[0] = buffer;
                            sorted = 0;
                        }
                    }
                }
            }
            timing(&startComm,rank);
            MPI_Barrier(MPI_COMM_WORLD);
            timing(&endComm,rank);
            aggrTiming(&comm, startComm , endComm, rank);
            //2 odd inner sort
            // innersort(nums, 1 ,count);
            if(rank<numtasks){
                for (i = 1; i < count-1; i += 2){
                    if (nums[i] > nums[i+1]){ 
                        swap(&nums[i], &nums[i+1]);
                        sorted = 0;
                    }
                }
            }
            int tmp = sorted;
            timing(&startComm,rank);
            MPI_Allreduce(&tmp, &sorted, 1, MPI_INT, MPI_BAND, MPI_COMM_WORLD);
            timing(&endComm,rank);
            aggrTiming(&comm, startComm , endComm, rank);
        }
        else{
            for (odd_even = 0; odd_even < 2; odd_even++){
                for (i = odd_even; i < count-1; i += 2){
                    if (nums[i] > nums[i+1]){ 
                        swap(&nums[i], &nums[i+1]);
                        sorted = 0;
                    }
                }	
            }
        }
    }
    timing(&startIo,rank);
    // write file
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh); 
	MPI_File_set_view(fh, sizeof(float) * rankSize * rank,  MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
	MPI_File_write(fh, nums, count, MPI_FLOAT, &status);
	MPI_File_close(&fh);
    timing(&endIo,rank);
    aggrTiming(&io,startIo,endIo,rank);
    free(nums);
    timing(&end, rank);
    aggrTiming(&latency, start,end,rank);
    if(rank==0)
        printf("CPU : %lf\nIO : %lf\nCommunication : %lf\nTotal : %lf\n" , (latency-io-comm)/CLOCKS_PER_SEC , io/CLOCKS_PER_SEC , comm/CLOCKS_PER_SEC , latency/CLOCKS_PER_SEC);
    MPI_Finalize();
}