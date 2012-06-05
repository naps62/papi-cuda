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

#ifndef _CAMERA_H
#define	_CAMERA_H

#include "utils.h"
#include "geometry/vector.h"
#include "ray.h"
#include "geometry/transform.h"
#include "display/film.h"
#include "path.h"
#include "pathtracer.h"

class PerspectiveCamera {
public:

	uint width, height;

	/* User defined values */
	Point orig, target;
	float fieldOfView;
	/* Calculated values */

	Vector dir, x, y;
	__HD__
	PerspectiveCamera() {

	}
	__HD__
	PerspectiveCamera(const Point &o, const Point &t, uint width_, uint height_) :
		orig(o), target(t), fieldOfView(45.f) {
		width = width_;
		height = height_;

		Update();
	}
	__HD__
	~PerspectiveCamera() {
	}
	__HD__
	void TranslateLeft(const float k) {
		Vector t = -k * Normalize(x);
		Translate(t);
	}
	__HD__
	void TranslateRight(const float k) {
		Vector t = k * Normalize(x);
		Translate(t);
	}
	__HD__
	void TranslateForward(const float k) {
		Vector t = k * dir;
		Translate(t);
	}
	__HD__
	void TranslateBackward(const float k) {
		Vector t = -k * dir;
		Translate(t);
	}
	__HD__
	void Translate(const Vector &t) {
		orig += t;
		target += t;
	}
	__HD__
	void RotateLeft(const float k) {
		Rotate(k, y);
	}
	__HD__
	void RotateRight(const float k) {
		Rotate(-k, y);
	}
	__HD__
	void RotateUp(const float k) {
		Rotate(k, x);
	}
	__HD__
	void RotateDown(const float k) {
		Rotate(-k, x);
	}
	__HD__
	void Rotate(const float angle, const Vector &axis) {
		Vector p = target - orig;
		Transform t = ::Rotate(angle, axis);
		target = orig + t(p);
	}
	__HD__
	void Update() {
		dir = target - orig;
		dir = Normalize(dir);

		const Vector up(0.f, 0.f, 1.f);

		const float k = Radians(fieldOfView);
		x = Cross(dir, up);
		x = Normalize(x);
		x *= width * k / height;

		y = Cross(x, dir);
		y = Normalize(y);
		y *= k;
	}

	__HD__
	void GenerateRay(Path *p, Ray *ray) const {
		const float cx = p->screenX / width - .5f;
		const float cy = p->screenY / height - .5f;
		Vector rdir = x * cx + y * cy + dir;
		Point rorig = orig;
		rorig += rdir * 0.1f;
		rdir = Normalize(rdir);

		*ray = Ray(rorig, rdir);
	}

};

#endif	/* _CAMERA_H */
