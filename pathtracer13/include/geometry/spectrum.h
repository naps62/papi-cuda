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

#ifndef LUX_RGB_H
#define LUX_RGB_H

#include <cmath>
#include <algorithm>
#include <ostream>



using namespace std;
using std::ostream;


#include "pathtracer.h"

class RGB {
public:
	// RGB Public Methods

	float r, g, b;

	__HD__
	RGB(float _r = 0.f, float _g = 0.f, float _b = 0.f)
	: r(_r), g(_g), b(_b) {
	}
	__HD__
	RGB operator+(const RGB &v) const {
		return RGB(r + v.r, g + v.g, b + v.b);
	}
	__HD__
	RGB operator*(const RGB &v) const {
		return RGB(r * v.r, g * v.g, b * v.b);
	}
	__HD__
	RGB & operator*=(const RGB &v) {
		r *= v.r;
		g *= v.g;
		b *= v.b;
		return *this;
	}
	__HD__
	RGB & operator+=(const RGB &v) {
		r += v.r;
		g += v.g;
		b += v.b;
		return *this;
	}
	__HD__
	RGB operator-(const RGB &v) const {
		return RGB(r - v.r, g - v.g, b - v.b);
	}
	__HD__
	RGB & operator-=(const RGB &v) {
		r -= v.r;
		g -= v.g;
		b -= v.b;
		return *this;
	}
	__HD__
	bool operator==(const RGB &v) const {
		return r == v.r && g == v.g && b == v.b;
	}
	__HD__
	RGB operator*(float f) const {
		return RGB(f*r, f*g, f * b);
	}
	__HD__
	RGB & operator*=(float f) {
		r *= f;
		g *= f;
		b *= f;
		return *this;
	}
	__HD__
	RGB operator/(float f) const {
		float inv = 1.f / f;
		return RGB(r * inv, g * inv, b * inv);
	}
	__HD__
	RGB & operator/=(float f) {
		float inv = 1.f / f;
		r *= inv;
		g *= inv;
		b *= inv;
		return *this;
	}
	__HD__
	RGB operator-() const {
		return RGB(-r, -g, -b);
	}
	__HD__
	float operator[](int i) const {
		return (&r)[i];
	}
	__HD__
	float &operator[](int i) {
		return (&r)[i];
	}
	__HD__
	float filter() const {
		return Max<float>(r, Max<float>(g, b));
	}
	__HD__
	bool Black() const {
		return (r == 0.f) && (g == 0.f) && (g == 0.f);
	}

};

inline ostream & operator<<(ostream &os, const RGB &v) {
	os << v.r << ", " << v.g << ", " << v.b;
	return os;
}
__HD__
inline RGB operator*(float f, const RGB &v) {
	return v * f;
}

#endif //LUX_Spectrum_H
