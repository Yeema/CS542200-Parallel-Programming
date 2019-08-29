#include <math.h>
#include <limits.h>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream> 
#include<assert.h>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atoi */
#include <mpi.h>
#include <string.h>
#include <pthread.h>

using namespace std;

#define V 2000000000
#define INF 2147483647
#define tag_ring_check 0
#define tag_terminate 1

#define NUM_THREADS 12
int n ,m, *edges, *Dist,numEdge;
pthread_mutex_t mutex[NUM_THREADS];
bool change;

void* Moore(void *threadId) {
    int* data = (int*) threadId;
    int* tmp = new int[n];
    //copy temp
    memcpy(tmp, Dist, sizeof(Dist[0])*n);
    for(int i=(*data);i<numEdge;i+=NUM_THREADS){
        int v1 = edges[3*i] , v2 = edges[3*i + 1] , weight = edges[3*i + 2];
        if (tmp[v1] != INF && tmp[v2] > tmp[v1] + weight){
            tmp[v2] = tmp[v1] + weight;
        }
        
    }
    // compare distance
    for(int i=0;i<NUM_THREADS;i++){
        int id = (i+(*data))%NUM_THREADS;
        pthread_mutex_lock(&mutex[id]);
        for(int j=id;j<n;j+=NUM_THREADS){          
            if (tmp[j] < Dist[j]){
                Dist[j] = tmp[j];
                change = true;
            }
        }
        pthread_mutex_unlock(&mutex[id]);
    }

    delete [] tmp;
    pthread_exit(NULL); 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    assert(argc == 3);
    int rank, procNum;
    int i, j;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    // read file
    ifstream infile(argv[1],ios::in | ios::binary);
    if (infile.is_open())
    {
        //size = infile.tellg();
        //memblock = new char [size];
        infile.seekg(0, ios::beg);
        infile.read((char*)&n, sizeof(n));
        infile.read((char*)&m, sizeof(m));
        Dist = new int[n];
        if(m%procNum>rank)
            numEdge = (int)(m/procNum)+1;
        else
            numEdge = (int)(m/procNum);
        edges = new int [3*numEdge];
        Dist[0] = 0;
        for (i=1;i<n;i++){
            Dist[i] = INF;
        }
        // cout<<n<<" "<<m<<"\n";
        // Triplet *tp = new Triplet();
        int _v1,_v2,_w;
        for (int i=0,j=0;i<m;i++){
            infile.read((char*)&_v1, sizeof(int));
            infile.read((char*)&_v2, sizeof(int));
            infile.read((char*)&_w, sizeof(int));
            if(i%procNum==rank){
                edges[3*j] = _v1;
                edges[3*j+1] = _v2;
                edges[3*j+2] = _w;
                j++;
            }
        }
        infile.close();
    }
    //prepare thread
    for(i =0;i<NUM_THREADS;i++){
        pthread_mutex_init (&mutex[i], NULL);
    }
    pthread_t threads[NUM_THREADS];
    int *td = (int*) malloc(NUM_THREADS*sizeof(int));
    for(i=0; i<NUM_THREADS; i++){
        td[i] = i;
    }
    //prepare MPI communication
    change = true;
    int *buffer = (int*) malloc(n*sizeof(int));
    MPI_Request rq;
    MPI_Status status;
    //Moore's algo
    while(change)
    {
        change = false;
        for(int i=0; i<NUM_THREADS; i++){
            pthread_create(&threads[i], NULL, Moore, (void*) (td+i));
        }
        for(int i=0; i<NUM_THREADS; i++){
            pthread_join(threads[i], NULL);
        }

        if (rank==0 && procNum>1){
            MPI_Send(Dist, n, MPI_INT, rank+1, tag_ring_check, MPI_COMM_WORLD);
            MPI_Recv(buffer, n, MPI_INT, procNum-1, tag_ring_check, MPI_COMM_WORLD, &status);
            for(i=1;i<procNum;i*=2){
                MPI_Send(buffer, n, MPI_INT,i, tag_terminate, MPI_COMM_WORLD);
            }
            if (!change){
                // check_diff(Dist, buffer, n);
                for(int i=0;i<n;i++){
                    if(buffer[i] != Dist[i]){
                        change = true;
                        break;
                    }
                }
            }
            swap(Dist, buffer);
        }
        else if (rank>0){
            MPI_Recv(buffer, n, MPI_INT, rank-1, tag_ring_check, MPI_COMM_WORLD, &status);
            // update Dist
            for(int i=0;i<n;i++){
                if(buffer[i] > Dist[i]){
                    buffer[i] = Dist[i];
                }
            }
            MPI_Send(buffer, n, MPI_INT, (rank+1)%procNum, tag_ring_check, MPI_COMM_WORLD);
            // send terminate
            for(i=2 ; rank/i>0;i*=2);
            // 0->1->3 0->2
            MPI_Recv(buffer, n, MPI_INT, rank-i/2, tag_terminate, MPI_COMM_WORLD, &status);
            for(;(rank+i)<procNum ;i*=2){
                MPI_Send(buffer, n, MPI_INT, rank+i, tag_terminate, MPI_COMM_WORLD);
            }
            if (!change){
                // check change
                for(int i=0;i<n;i++){
                    if(buffer[i] != Dist[i]){
                        change = true;
                        break;
                    }
                }
            }
            swap(Dist, buffer);
        }
    }

    for(i =0;i<NUM_THREADS;i++){
        pthread_mutex_destroy(&mutex[i]);
    }
    //wrtie file
    if(rank==0){
        ofstream ofile;
        ofile.open(argv[2], ios::binary | ios::out);
        for(int i=0; i<n ;i++){
            ofile.write((char*) &Dist[i], sizeof(int));
        }
        ofile.close();
    }

    MPI_Finalize();
    delete [] Dist;
    delete [] edges;
    free(td);
    free(buffer);
    return 0;
}
