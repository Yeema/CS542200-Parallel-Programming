#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>
#include <math.h>
#include <time.h>
int main(int argc, char* argv[])
{
    int niter = 1000000;                   
    int myid;                       
    double x,y;                     
    int i;                          
    int count=0;                
    double z;                       
    double pi;                      
    int reducedcount;                   
    int reducedniter;                   
    int ranknum = 0;                    
    int numthreads = 16;
    MPI_Init(&argc, &argv);                 
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);           
    MPI_Comm_size(MPI_COMM_WORLD, &ranknum);        
 
    if(myid != 0)                       //Do the following on all except the master node
    {
        //Start OpenMP code: 16 threads/node
        #pragma omp parallel firstprivate(x, y, z, i) reduction(+:count) num_threads(numthreads)
        {
            srandom((int)time(NULL) ^ omp_get_thread_num());    //Give random() a seed value
            for (i=0; i<niter; ++i)              //main loop
            {
                x = (double)random()/RAND_MAX;      
                y = (double)random()/RAND_MAX;      
                z = sqrt((x*x)+(y*y));          
                if (z<=1)
                {
                    ++count;            
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&count,
                   &reducedcount,
                   1,
                   MPI_INT,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);
    reducedniter = numthreads*niter*(ranknum-1);
 
    if (myid == 0)                      //if root process/master node
    {
            //p = 4(m/n)
        pi = ((double)reducedcount/(double)reducedniter)*4.0;
        //Print the calculated value of pi
        printf("Pi: %f\n%i\n%d\n", pi, reducedcount, reducedniter);
    }
 
    MPI_Finalize();                     //Close the MPI instance
    return 0;
}