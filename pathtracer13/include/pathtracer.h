/*
 * pathtracer.h
 *
 *  Created on: Feb 6, 2012
 *      Author: rr
 */

#ifndef PATHTRACER_H_
#define PATHTRACER_H_


#if defined(__CUDACC__)
#define __HD__ 			__device__
#define __H_D__			__host__ __device__
#define __noinline 		__noinline__
#define __forceinline 	__forceinline__

#else
#define __HD__
#define __H_D__
#define __noinline
#define __forceinline 	__inline__ __attribute__((__always_inline__))

#endif


#if !defined(__CUDACC__)

#include <stdio.h>
#include <stdlib.h>
//#include <iostream>
#include <sys/time.h>

#include "config.h"


class PERFORM_acc_timer {
public:

	timeval t;
	timeval global_start;

	double duration;


	PERFORM_acc_timer() {
		duration=0;
	}

	~PERFORM_acc_timer() {
	}

	void start() {
		gettimeofday(&t, 0);
	}

	void stop() {
		timeval end;
		gettimeofday(&end, 0);
		duration += get_interval(t, end) / 1000.0; //seconds
	}

	/**
	 * returns ms
	 */
	static double get_interval(timeval start, timeval end) {
		double v = (double) (end.tv_sec * 1000.0 + end.tv_usec / 1000.0 - start.tv_sec * 1000.0 - start.tv_usec / 1000.0 + 0.5);
		return v;

	}


	char*  print() {
		char* buffer= new char[32];
		snprintf(buffer, 32, "%g", duration);
		return buffer;
	}

};

#endif

typedef unsigned int uint;

typedef struct {
	unsigned long z1, z2, z3, z4;
} Seed;

#endif /* PATHTRACER_H_ */
