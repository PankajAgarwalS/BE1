#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void swap();

void *func(void *p) {
      long tid;
      tid = (long)p;
      //printf("From the function, the thread id = %ld", pthread_self()); //get current thread id
      printf("ID: %ld", pthread_self());
      pthread_exit(NULL);
}

int main (int argc, char *argv[]) {
	long SIZE =1<<8;
	long A[SIZE];

	for(long i=0;i<SIZE;i++)
	{
	    A[i]=rand()%SIZE;
	}
	//int A[5] = {6,9,1,3,7};
	long N = SIZE;
    long i=0;
	long j=0; 
	long first;
	double start,end;
	int rc;
    pthread_t thread[SIZE];
	start=omp_get_wtime();
	for( i = 0; i < N-1; i++ )
	{
		first = i % 2; 
		#pragma omp parallel for default(none),shared(A,first,N)
		for( j = first; j < N-1; j += 1 )
		{
			if( A[ j ] > A[ j+1 ] )
			{
				swap( &A[ j ], &A[ j+1 ] );
			}
		}
	}
	end=omp_get_wtime();
	for(i=0;i<N;i++)
	{
                rc = pthread_create(&thread[i], NULL, func, (void *)i);
                //printf("From the main function, the thread id = %ld", thread);
                //pthread_join(thread, NULL); //join with main thread
				printf(" %ld\n",A[i]);
	}

	printf("\n-------------------------\n Time Parallel= %f\n",(end-start));
}

void swap(int *num1, int *num2)
{

	long temp = *num1;
	*num1 =  *num2;
	*num2 = temp;
}
