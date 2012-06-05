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

#ifndef _PATH_H
#define	_PATH_H

#include "ray.h"
#include "geometry/spectrum.h"
#include "pathtracer.h"

class prePath {
public:

	float screenX, screenY;
	//Seed seed;

	__HD__
	prePath() {
		screenX = 0.f;
		screenY = 0.f;
		//seed.z1 = 0ul;
		//seed.z2 = 0ul;
		//seed.z3 = 0ul;
		//seed.z4 = 0ul;


	}
	__HD__
	~prePath() {
	}

};


class Path {
public:

	float screenX, screenY;
	RGB throughput;
	RGB radiance;
	int depth;
	Ray pathRay;
	Seed seed;

	__HD__
	Path(prePath* p) {

		screenX = p->screenX;
		screenY = p->screenY;

		throughput = RGB(1.f, 1.f, 1.f);
		radiance = RGB(0.f, 0.f, 0.f);
		depth = 0;

	}

	__HD__
	Path() {

		throughput = RGB(1.f, 1.f, 1.f);
		radiance = RGB(0.f, 0.f, 0.f);
		depth = 0;

	}

	__HD__
	~Path() {

	}

};

#endif	/* _PATH_H */
