#include <stdio.h>
#include<math.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
double sum=0.0;
int num_threads;
int size, error;
void* calculate(void* threadid){
	int* tid = (int*)threadid;
    int i;
    int rank = *tid;
    double x2;
    double result=0.0;
    int N=1000000;
    for (i=rank; i<N; i+=num_threads)
    {
        x2=(double)i*(double)i/((double)N*(double)N);
        result+=sqrt(1-x2)/N;
    }
    pthread_mutex_lock(&mutex);
    sum+=result;
    pthread_mutex_unlock(&mutex);
    pthread_exit(NULL);
}
int main (int argc, char* argv[])
{
    double pi=0.0;
    num_threads = atoi(argv[1]);
    // size = atoi(argv[2]);
    pthread_t threads[num_threads];
    int ID[num_threads];
    int t;
    for (t = 0; t < num_threads; t++) {
        ID[t] = t;
        int rc = pthread_create(&threads[t], NULL, calculate, (void*)&ID[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    for(t=0;t<num_threads;t++)
        pthread_join(threads[t], NULL);
    pi=4*sum;
    printf("%f",pi);
    
    return 0;
}
