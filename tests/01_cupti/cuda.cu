#include <cuda.h>
#include <cupti.h>
#include <stdio.h>

#define N 10
#define THREADS 11

#define EVENT_NAME_TESLA "instructions"
#define EVENT_NAME_FERMI "inst_executed"

#define CHECK_CU_ERROR(err, cufunc)										\
	if (err != CUDA_SUCCESS) { 											\
		printf("%s:%d: error %d for CUDA Driver API function '%s'\n",	\
				__FILE__, __LINE__, err, cufunc);						\
		exit(-1);														\
	}

#define CHECK_CUPTI_ERROR(err, cuptifunc)								\
	if (err != CUPTI_SUCCESS) {											\
		const char *errstr;												\
		cuptiGetResultString(err, &errstr);								\
		printf("%s:%d:Error %s for CUPTI API function '%s'\n",			\
				__FILE__, __LINE__, errstr, cuptifunc);					\
		exit(-1);														\
	}

typedef struct cupti_eventData_st {
	CUpti_EventGroup eventGroup;
	CUpti_EventID eventId;
} cupti_eventData;

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
	cupti_eventData *eventData;
	uint64_t eventVal;
} RuntimeApiTrace_t;

void CUPTIAPI getEventValueCallback(
						void *userdata,
						CUpti_CallbackDomain domain,
						CUpti_CallbackId cbid,
						const CUpti_CallbackData *cbInfo) {

	CUptiResult cuptiErr;
	RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*) userdata;
	size_t bytesRead;

	// This callback is enabled for launch so we shouldn't see anything else.
	if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
		printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
		exit(-1);
	}

	switch(cbInfo->callbackSite) {
		case CUPTI_API_ENTER:
			cudaThreadSynchronize();
			cuptiErr = cuptiSetEventCollectionMode(cbInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL);
			CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");
			cuptiErr = cuptiEventGroupEnable(traceData->eventData->eventGroup);
			CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");
			break;

		case CUPTI_API_EXIT:
			bytesRead = sizeof(uint64_t);
			cudaThreadSynchronize();
			cuptiErr = cuptiEventGroupReadEvent(traceData->eventData->eventGroup, CUPTI_EVENT_READ_FLAG_NONE, traceData->eventData->eventId, &bytesRead, &traceData->eventVal);
			CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
			cuptiErr = cuptiEventGroupDisable(traceData->eventData->eventGroup);
			CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");
			break;
	}
}

static void displayEventVal(RuntimeApiTrace_t *trace, char *eventName) {
	printf("Event Name: %s \n", eventName);
	printf("Event Value: %llu\n", (unsigned long long) trace->eventVal);
}

__global__ void kernel(int *arr) {

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N) return;

	arr[id] = 1;
}

int main() {
	int deviceCount;
	CUcontext context = 0;
	CUdevice dev = 0;
	char deviceName[32];
	char *eventName;
	CUptiResult cuptiErr;
	CUpti_SubscriberHandle subscriber;
	cupti_eventData cuptiEvent;
	RuntimeApiTrace_t trace;
	int cap_major, cap_minor;

	CUresult err = cuInit(0);
	CHECK_CU_ERROR(err, "cuInit");

	err = cuDeviceGetCount(&deviceCount);
	CHECK_CU_ERROR(err, "cuDeviceGetCount");

	if (deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
		return -2;
	}

	err = cuDeviceGet(&dev, 0);
	CHECK_CU_ERROR(err, "cuDeviceGet");

	err = cuDeviceGetName(deviceName, 32, dev);
	CHECK_CU_ERROR(err, "cuDeviceGetName");

	err = cuDeviceComputeCapability(&cap_major, &cap_minor, dev);
	CHECK_CU_ERROR(err, "cuDeviceComputeCapability");

	printf("CUDA Device Name: %s\n", deviceName);
	printf("CUDA Capability: %d.%d\n", cap_major, cap_minor);

	err = cuCtxCreate(&context, 0, dev);
	CHECK_CU_ERROR(err, "cuCtxCreate");

	cuptiErr = cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");

	if (cap_major > 1)
		eventName = EVENT_NAME_FERMI;
	else
		eventName = EVENT_NAME_TESLA;

	cuptiErr = cuptiEventGetIdFromName(dev, eventName, &cuptiEvent.eventId);
	if (cuptiErr != CUPTI_SUCCESS) {
		printf("Invalid eventName: %s\n", eventName);
		return -1;
	}

	cuptiErr = cuptiEventGroupAddEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

	trace.eventData = &cuptiEvent;

	cuptiErr = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getEventValueCallback, &trace);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiSubscribe");

	cuptiErr = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEnableCallback");


	int host_arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int *dev_arr;
	

	cudaMalloc(&dev_arr, sizeof(int) * N);
	cudaMemcpy(dev_arr, &host_arr, sizeof(int) * N, cudaMemcpyHostToDevice);
	kernel<<< THREADS, 1 >>>(dev_arr);

	displayEventVal(&trace, eventName);
	trace.eventData = NULL;

	cuptiErr = cuptiEventGroupRemoveEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupRemoveEvent");

	cuptiErr = cuptiEventGroupDestroy(cuptiEvent.eventGroup);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");

	cuptiErr = cuptiUnsubscribe(subscriber);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiUnsubscribe");

	cudaDeviceSynchronize();
}
