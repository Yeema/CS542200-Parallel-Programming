#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mpi.h"
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream> 
#include<assert.h>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atoi */
#include <mpi.h>
#include <string.h>
using namespace std;

#define V 2000000000
// #define INF 2147483647
#define INF 1147483647

int n; // Number of vertices
unsigned int m;	// Number of edges


int main(int argc, char *argv[]){
    int size, rank;
    assert(argc == 3);
    MPI_Init(&argc,&argv);
    double time_start = MPI_Wtime();
    MPI_Datatype rtype;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream infile(argv[1],ios::in | ios::binary);
    if (infile.is_open())
    {
        infile.seekg(0, ios::beg);
        infile.read((char*)&n, sizeof(n));
        infile.read((char*)&m, sizeof(m));
    }
    int* Dist = (int*) malloc(n*n * sizeof(int));
    int* result = (int*) malloc(n*n * sizeof(int));
    #pragma omp parallel for schedule(dynamic)
    for(int i=rank;i<n;i+=size){
        #pragma omp parallel for schedule(dynamic)
        for (int j=0;j<n;j++)
        {
            Dist[i*n+j] = INF;
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for(int i=rank;i<n;i+=size){
        Dist[i*n + i]=0;
    }
    int v1,v2,w;
    for (int i=0;i<m;i++){
        infile.read((char*)&v1, sizeof(v1));
        infile.read((char*)&v2, sizeof(v2));
        infile.read((char*)&w, sizeof(w));
        Dist[v1*n +v2] = w;
    }
    infile.close();

    int avg = n/size;
    int rowk;
    for(int k=0; k< n ; k++){
        rowk = k%size;
        MPI_Bcast(Dist+k*n, n, MPI_INT,rowk, MPI_COMM_WORLD);
        // MPI_Bcast(Dist[k], n, MPI_INT,rowk, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(dynamic)
        for(int i=rank;i<n;i+=size){
            if(Dist[i*n + k]!=INF){
                #pragma omp parallel for schedule(dynamic)
                for(int j = 0; j<n ; j++){
                    if(Dist[k*n + j]!=INF && Dist[i*n + j] > Dist[i*n + k]+Dist[k*n + j]){
                            Dist[i*n + j] = Dist[i*n + k]+Dist[k*n + j];
                    }
                }
            }
        }
    }
    MPI_Reduce(Dist,result,n*n,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    
    // /*              write file           */
    if(rank==0){
        ofstream ofile;
        ofile.open(argv[2], ios::binary | ios::out);
        for(int i=0; i<n ;i++){
            for(int j=0 ; j<n ; j++){
                ofile.write((char*) &result[i*n+j], sizeof(int));
            }
        }
        ofile.close();
    }
    double time_end = MPI_Wtime();
    cout<<"Rank "<<rank<<" Time: "<<(time_end-time_start);
    MPI_Finalize();
    free(Dist);
    free(result);
    
    return 0;
}