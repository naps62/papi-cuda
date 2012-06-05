/***************************************************************************
 *   Copyright (C) 1998-2009 by authors (see AUTHORS.txt )                 *
 *                                                                         *
 *   This file is part of LuxRender.                                       *
 *                                                                         *
 *   Lux Renderer is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   Lux Renderer is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 *   This project is based on PBRT ; see http://www.pbrt.org               *
 *   Lux Renderer website : http://www.luxrender.net                       *
 ***************************************************************************/

#ifndef LUX_POINT_H
#define LUX_POINT_H

#include "vector.h"

#include "pathtracer.h"

class Point {
public:
	// Point Methods
	__HD__
	Point(float _x = 0, float _y = 0, float _z = 0)
	: x(_x), y(_y), z(_z) {
	}
	__HD__
	Point(float v[3]) : x(v[0]), y(v[1]), z(v[2]) {
	}
	__HD__
	Point operator+(const Vector &v) const {
		return Point(x + v.x, y + v.y, z + v.z);
	}
	__HD__
	Point & operator+=(const Vector &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	__HD__
	Vector operator-(const Point &p) const {
		return Vector(x - p.x, y - p.y, z - p.z);
	}
	__HD__
	Point operator-(const Vector &v) const {
		return Point(x - v.x, y - v.y, z - v.z);
	}
	__HD__
	Point & operator-=(const Vector &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	__HD__
	Point & operator+=(const Point &p) {
		x += p.x;
		y += p.y;
		z += p.z;
		return *this;
	}
	__HD__
	Point & operator-=(const Point &p) {
		x -= p.x;
		y -= p.y;
		z -= p.z;
		return *this;
	}
	__HD__
	Point operator+(const Point &p) const {
		return Point(x + p.x, y + p.y, z + p.z);
	}
	__HD__
	Point operator*(float f) const {
		return Point(f*x, f*y, f * z);
	}
	__HD__
	Point & operator*=(float f) {
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	__HD__
	Point operator/(float f) const {
		float inv = 1.f / f;
		return Point(inv*x, inv*y, inv * z);
	}
	__HD__
	Point & operator/=(float f) {
		float inv = 1.f / f;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}
	__HD__
	float operator[](int i) const {
		return (&x)[i];
	}
	__HD__
	float &operator[](int i) {
		return (&x)[i];
	}
	// Point Public Data
	float x, y, z;
};
__HD__
inline Vector::Vector(const Point &p)
: x(p.x), y(p.y), z(p.z) {
}
__HD__
inline ostream & operator<<(ostream &os, const Point &v) {
	os << v.x << ", " << v.y << ", " << v.z;
	return os;
}
__HD__
inline Point operator*(float f, const Point &p) {
	return p*f;
}
__HD__
inline float Distance(const Point &p1, const Point &p2) {
	return (p1 - p2).Length();
}
__HD__
inline float DistanceSquared(const Point &p1, const Point &p2) {
	return (p1 - p2).LengthSquared();
}

#endif //LUX_POINT_H
