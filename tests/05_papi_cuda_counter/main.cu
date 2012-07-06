#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <papi.h>

#define PAPI_ERROR(n,v) (fprintf(stderr, "%s failed with code %d\n", (n), (v)))

#include "../common/kernels.cu"


__host__
int main (int argc, char * argv[]) {
	int retval;

	//	arguments
	int version = atoi(argv[1]);
	int nthreads = atoi(argv[2]);
	int nblocks = atoi(argv[3]);
	int N = nthreads;



	//	Initialize library
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT) {
		PAPI_ERROR("PAPI_library_init", retval);
	}
	fprintf(stderr, "PAPI version %4d %6d %7d\n", PAPI_VERSION_MAJOR(PAPI_VERSION), PAPI_VERSION_MINOR(PAPI_VERSION), PAPI_VERSION_REVISION(PAPI_VERSION));



	int eventcnt = 0;



	int tmp_eventcnt = argc - 4;
	char ** tmp_argv = argv + 4;
	char ** tmp_names = new char*[tmp_eventcnt];
	int * tmp_events = new int[tmp_eventcnt];
	for (int i = 0; i < tmp_eventcnt; ++i) {
		fprintf(stderr, "%s\n", tmp_argv[i]);
		retval = PAPI_event_name_to_code(tmp_argv[i], tmp_events + eventcnt);
		if (retval != PAPI_OK)
			PAPI_ERROR("PAPI_event_name_to_code", retval);
		else {
			fprintf(stderr, "Event \"%s\" --- Code: %x\n", tmp_argv[i], tmp_names[eventcnt]);
			tmp_names[eventcnt++] = tmp_argv[i];
		}
	}



	char ** names = new char*[eventcnt];
	int * events = new int[eventcnt];
	long long int * values = new long long int[eventcnt];
	for (int i = 0 ; i < eventcnt; ++i) {
		names[i] = tmp_names[i];
		events[i] = tmp_events[i];
	}
	memset(values, 0, sizeof(long long int));



	free(tmp_names);
	free(tmp_events);


	int *host_arr = (int *) malloc(sizeof(int) * N);
	int *dev_arr;

	srand(time(NULL));
	for(int i = 0; i < N; ++i)
		host_arr[i] = rand();
	

	cudaMalloc(&dev_arr, sizeof(int) * N);
	cudaMemcpy(dev_arr, &host_arr, sizeof(int) * N, cudaMemcpyHostToDevice);



	int set = PAPI_NULL;
	retval = PAPI_create_eventset(&set);
	if (retval != PAPI_OK)
		PAPI_ERROR("PAPI_create_eventset", retval);

	retval = PAPI_add_events(set, events, eventcnt);
	if (retval != PAPI_OK)
		PAPI_ERROR("PAPI_add_events", retval);



	retval = PAPI_start(set);
	if (retval != PAPI_OK)
		PAPI_ERROR("PAPI_start", retval);

	if (version == 1)
		kernel_one<<< nthreads, nblocks >>>(dev_arr, N);
	else
		kernel_two<<< nthreads, nblocks >>>(dev_arr, N);


	retval = PAPI_stop(set, values);
	if (retval != PAPI_OK)
		PAPI_ERROR("PAPI_stop", retval);
	
	for (int i = 0; i < eventcnt; ++i)
		printf("%s\t%x\t%lld\n", names[i], events[i], values[i]);



	retval = PAPI_cleanup_eventset(set);
	if (retval != PAPI_OK)
		PAPI_ERROR("PAPI_cleanup_eventset", retval);

	retval = PAPI_destroy_eventset(&set);
	if (retval != PAPI_OK)
		PAPI_ERROR("PAPI_destroy_eventset", retval);



	free(names);
	free(events);
	free(values);



	//	Shutdown library
	PAPI_shutdown();

	return 0;
}
