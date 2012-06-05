/***************************************************************************
 *   Copyright (C) 1998-2009 by David Bucciarelli (davibu@interfree.it)    *
 *                                                                         *
 *   This file is part of SmallLuxGPU.                                     *
 *                                                                         *
 *   SmallLuxGPU is free software; you can redistribute it and/or modify   *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *  SmallLuxGPU is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 *   This project is based on PBRT ; see http://www.pbrt.org               *
 *   and Lux Renderer website : http://www.luxrender.net                   *
 ***************************************************************************/

#ifndef _SMALLLUX_H
#define	_SMALLLUX_H

#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>



#include <stddef.h>
#include <sys/time.h>

#include "pathtracer.h"

#define __NO_STD_VECTOR
#define __NO_STD_STRING

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef INFINITY
#define INFINITY numeric_limits<float>::infinity()
#endif

#ifndef INV_PI
#define INV_PI  0.31830988618379067154f
#endif

__HD__
__inline__ void swap(float* x, float* y) {
	float t;
	t = *y;
	*y = *x;
	*x = t;
}

template<class T>
__HD__
inline T Max(T a, T b) {
		return a > b ? a : b;
	}

template<class T>
__HD__
inline T Min(T a, T b) {
		return a < b ? a : b;
	}

inline double WallClockTime() {
#if defined(__linux__) || defined(__APPLE__)
	struct timeval t;
	gettimeofday(&t, NULL);

	return t.tv_sec + t.tv_usec / 1000000.0;
#elif defined (WIN32)
	return GetTickCount() / 1000.0;
#else
	Unsupported Platform !!!
#endif
}


inline float Radians(float deg) {
	return (M_PI / 180.f) * deg;
}

inline float Degrees(float rad) {
	return (180.f / M_PI) * rad;
}

template<class T> inline T Clamp(T val, T low, T high) {
	return val > low ? (val < high ? val : high) : low;
}

template<class T> inline int Float2Int(T val) {
	return static_cast<int> (val);
}

template<class T> inline unsigned int Float2UInt(T val) {
	return val >= 0 ? static_cast<unsigned int> (val) : 0;
}

inline int Floor2Int(double val) {
	return static_cast<int> (floor(val));
}

inline int Floor2Int(float val) {
	return static_cast<int> (floorf(val));
}

inline unsigned int Floor2UInt(double val) {
	return val > 0. ? static_cast<unsigned int> (floor(val)) : 0;
}
__HD__
inline unsigned int Floor2UInt(float val) {
	return val > 0.f ? static_cast<unsigned int> (floorf(val)) : 0;
}

inline int Ceil2Int(double val) {
	return static_cast<int> (ceil(val));
}

inline int Ceil2Int(float val) {
	return static_cast<int> (ceilf(val));
}

inline unsigned int Ceil2UInt(double val) {
	return val > 0. ? static_cast<unsigned int> (ceil(val)) : 0;
}

inline unsigned int Ceil2UInt(float val) {
	return val > 0.f ? static_cast<unsigned int> (ceilf(val)) : 0;
}

template <class T>
inline std::string ToString(const T& t) {
	std::stringstream ss;
	ss << t;
	return ss.str();
}

inline float PowerHeuristic(unsigned int nf, float fPdf, unsigned int ng, float gPdf) {
	const float f = nf * fPdf, g = ng * gPdf;
	return (f * f) / (f * f + g * g);
}

inline string ReadSources(const string &name, const string &fileName) {
	fstream file;
	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);
	file.open(fileName.c_str(), fstream::in | fstream::binary);

	string prog(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
	cerr << "[Kernel::" << name << "] Kernel file size " << prog.length() << "bytes" << endl;

	return prog;
}

#endif	/* _SMALLLUX_H */
