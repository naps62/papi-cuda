#ifndef ___PAPI_H___
#define ___PAPI_H___

#include <cupti.h>

#define CHECK_CUPTI_ERROR(err, cuptifunc)	\
	if (err != CUPTI_SUCCESS) {	\
		const char * errstr;	\
		cuptiGetResultString(err, &errstr);	\
		printf("%s:%d:Error %s for CUPTI API function \"%s\"\n", __FILE__, __LINE__, err, cuptifunc);	\
		exit(-1);	\
	}

typedef struct cupti_eventData_st {
	CUpti_EventGroup eventGroup;
	CUpti_EventID eventId;
} cupti_eventData;

typedef struct RuntimeApiTrace_st {
	cupti_eventData * eventData;
	uint64_t eventVal;
} RuntimeApiTrace_t;

void CUPTIAPI getEventValueCallback(void * userData, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData * cbInfo);

void papiTestInit(int argc, char * argv[]);
void papiTestCleanup();

#endif//___PAPI_H___
