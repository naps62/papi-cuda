#include <cstdio>

#include <papi.h>

#define PAPI_ERROR(n,v) (fprintf(stderr, "%s failed with code %d\n", (n), (v)))



__global__
void kernel () {
	//int blockX = blockIdx.x;
	//int blockY = blockIdx.y;

	//int threadX = threadIdx.x;
	//int threadY = threadIdx.y;
	//int threadZ = threadIdx.z;

	//cuPrintf("[%d,%d]\t[%d,%d,%d]\n", blockX, blockY, threadX, threadY, threadZ);
}



__host__
int main (int argc, char * argv[]) {
	int retval;

	//	arguments
	int nthreads = atoi(argv[1]);
	int nblocks = atoi(argv[2]);



	//	Initialize library
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT) {
		PAPI_ERROR("PAPI_library_init", retval);
	}
	fprintf(stderr, "PAPI version %4d %6d %7d\n", PAPI_VERSION_MAJOR(PAPI_VERSION), PAPI_VERSION_MINOR(PAPI_VERSION), PAPI_VERSION_REVISION(PAPI_VERSION));



	int eventcnt = 0;



	int tmp_eventcnt = argc - 3;
	char ** tmp_argv = argv + 3;
	char ** tmp_names = new char*[tmp_eventcnt];
	int * tmp_events = new int[tmp_eventcnt];
	for (int i = 0; i < tmp_eventcnt; ++i) {
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


	//cudaPrintfInit();
	kernel<<< nblocks, nthreads >>>();
	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();




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
