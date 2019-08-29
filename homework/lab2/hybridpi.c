#include <stdio.h>
#include<math.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
int main (int argc, char* argv[])
{
    int rank, size, error, i;
    double pi=0.0, result=0.0, sum=0.0, x2;
    
    error=MPI_Init (&argc, &argv);
    int N = atoi (argv[1]);
    //Get process ID
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    //Get processes Number
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    
    //Synchronize all processes and get the begin time
    // MPI_Barrier(MPI_COMM_WORLD);
    //Each process caculates a part of the sum
    #pragma omp parallel for reduction(+:result) private(x2)
    for (i=rank; i<N; i+=size)
    {
        x2=(double)i*(double)i/((double)N*(double)N);
        result+=sqrt(1-x2)/N;
    }
    
    //Sum up all results
    MPI_Reduce(&result, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    //Synchronize all processes and get the end time
    MPI_Barrier(MPI_COMM_WORLD);
    //end = MPI_Wtime();
    
     //Caculate and print PI
    if (rank==0)
    {
        pi=4*sum;
        printf("%f",pi);
    }
    
    error=MPI_Finalize();
    
    return 0;
}
