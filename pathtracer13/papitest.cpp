#include <cstdio>
#include <papi.h>

#include "papitest.h"

/*
typedef struct cupti_eventData_st {
	CUpti_EventGroup eventGroup;
	CUpti_EventID eventId;
} cupti_eventData;

typedef struct RuntimeApiTrace_st {
	cupti_eventData * eventData;
	uint64_t eventVal;
} RuntimeApiTrace_t;
*/

int eventcnt;
char ** names;
int * events;
long long int * values;
int set;

void CUPTIAPI getEventValueCallback(void * userData, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData * cbInfo) {
	CUptiResult cuptiErr;
	RuntimeApiTrace_t * traceData = (RuntimeApiTrace_t *) userData;
	size_t bytesRead;

	if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
		printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
		exit(-1);
	}

	int retval;
	switch (cbInfo->callbackSite) {
		case CUPTI_API_ENTER:
			// start PAPI
			if (eventcnt) {
				retval = PAPI_start(set);
				if (retval != PAPI_OK)
					fprintf(stderr, "PAPI_start failed with code %d\n", retval);
			}
			break;
		case CUPTI_API_EXIT:
			// stop PAPI
			if (eventcnt) {
				retval = PAPI_stop(set, values);
				if (retval != PAPI_OK)
					fprintf(stderr, "PAPI_stop failed with code %d\n", retval);
				printf("VALUES:\n");
			}
			for (int i = 0; i < eventcnt; ++i)
				printf("\t%s\t%x\t%lld\n", names[i], events[i], values[i]);
			break;
	}
}


void papiTestInit(int argc, char * argv[]) {
	int retval;

	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT)
		fprintf(stderr, "PAPI_library_init failed with code %d\n", retval);
	
	fprintf(stderr, "PAPI version: %4d %6d %7d\n", PAPI_VERSION_MAJOR(PAPI_VERSION), PAPI_VERSION_MINOR(PAPI_VERSION), PAPI_VERSION_REVISION(PAPI_VERSION));

	int * tmp_events = new int[argc];
	char ** tmp_names = new char*[argc];
	int i;
	eventcnt = 0;
	for (i = 0; i < argc; ++i) {
		retval = PAPI_event_name_to_code(argv[i], tmp_events + eventcnt);
		if (retval != PAPI_OK)
			fprintf(stderr, "PAPI_event_name_to_code failed with code %d\n", retval);
		else {
			fprintf(stderr, "Event \"%s\" --- Code: %x\n", argv[i], tmp_events[eventcnt]);
			tmp_names[eventcnt++] = argv[i];
		}
			
	}

	names = new char*[eventcnt];
	events = new int[eventcnt];
	values = new long long int[eventcnt];

	for (i = 0; i < eventcnt; ++i) {
		names[i] = tmp_names[i];
		events[i] = tmp_events[i];
	}
	memset(values, 0, sizeof(long long int));

	free(tmp_names);
	free(tmp_events);

	set = PAPI_NULL;
	retval = PAPI_create_eventset(&set);
	if (retval != PAPI_OK)
		fprintf(stderr, "PAPI_create_eventset failed with code %d\n", retval);

	retval = PAPI_add_events(set, events, eventcnt);
	if (retval != PAPI_OK)
		fprintf(stderr, "PAPI_add_events failed with code %d\n", retval);
}

void papiTestCleanup() {
	free(values);
	free(events);
	PAPI_shutdown();
}
