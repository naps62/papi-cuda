/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    HelloWorld.c
 * CVS:     $Id$
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	<your name here>
 *			<your email address>
 * test case for Example component 
 * 
 *
 * @brief
 *  This file is a very simple HelloWorld C example which serves (together
 *	with its Makefile) as a guideline on how to add tests to components.
 *  The papi configure and papi Makefile will take care of the compilation
 *	of the component tests (if all tests are added to a directory named
 *	'tests' in the specific component dir).
 *	See components/README for more details.
 *
 *	The string "Hello World!" is mangled and then restored.
 */

#include <cuda.h>
#include <stdio.h>
#include <papi.h>

#define NUM_EVENTS 1
#define PAPI

// Prototypes
__global__ void helloWorld(char*);


// Host function
int main(int argc, char** argv)
{
#ifdef PAPI
	int retval, i;
	int EventSet = PAPI_NULL;
	long long values[NUM_EVENTS];
	/* REPLACE THE EVENT NAME 'PAPI_FP_OPS' WITH A CUDA EVENT 
	   FOR THE CUDA DEVICE YOU ARE RUNNING ON.
	   RUN papi_native_avail to get a list of CUDA events that are 
	   supported on your machine */
    char *EventName[] = { "PAPI_FP_OPS" };
	int events[NUM_EVENTS];
	
	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if( retval != PAPI_VER_CURRENT )
		fprintf( stderr, "PAPI_library_init failed\n" );
	
	printf( "PAPI_VERSION     : %4d %6d %7d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	
	/* convert PAPI native events to PAPI code */
	for( i = 0; i < NUM_EVENTS; i++ ){
		retval = PAPI_event_name_to_code( EventName[i], &events[i] );
		if( retval != PAPI_OK )
			fprintf( stderr, "PAPI_event_name_to_code failed\n" );
		else
			printf( "Name %s --- Code: %x\n", EventName[i], events[i] );
	}

	retval = PAPI_create_eventset( &EventSet );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_create_eventset failed\n" );
	
	retval = PAPI_add_events( EventSet, events, NUM_EVENTS );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_add_events failed\n" );
	
	retval = PAPI_start( EventSet );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_start failed\n" );
#endif


	#define N 10

	int j;
	
	// desired output
	int str[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

	// mangle contents of output
	// the null character is left intact for simplicity
	for(j = 0; j < 12; j++) {
		str[j] -= j;
		//printf("str=%s\n", str);
	}

	if (argc != 4) {
		printf("Usage: ./a.out <threads> <blocks> <counter_name>")
		return -2;
	}

	int threads = atoi(argv[1]);
	int blocks = atoi(argv[2]);
	
	// allocate memory on the device
	char *d_str;

	cudaMalloc((void**)&d_str, sizeof(int) * N);
	
	// copy the string to the device
	cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);
	

	// invoke the kernel
	kernel<<< threads, blocks >>>(d_str);

	// retrieve the results from the device
	cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);
	
	// free up the allocated memory on the device
	cudaFree(d_str);
	
	printf("END: %s\n", str);

	
#ifdef PAPI
	retval = PAPI_stop( EventSet, values );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_stop failed\n" );

	for( i = 0; i < NUM_EVENTS; i++ )
		printf( "%12lld \t\t --> %s \n", values[i], EventName[i] );
#endif

	return 0;
}


// Device kernel
__global__ void kernel(int *arr) {
	// should return 11
	__prof_trigger(00);

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N) return;

	// should yield 10
	__prof_trigger(01);

	if (arr[id] < 4) __prof_trigger(02); //should yield 4
	else             __prof_trigger(03); //should yield 6
}