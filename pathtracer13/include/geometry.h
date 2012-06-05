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

#ifndef _PLYOBJECT_H
#define	_PLYOBJECT_H

#include <string.h>

#include "geometry/point.h"
#include "geometry/normal.h"
#include "geometry/spectrum.h"
#include "geometry/triangle.h"
#include "geometry/rply.h"
#include "geometry/light.h"

#include "utils.h"



class Geometry {
public:

	Geometry();
	Geometry(const string &fileName);
	Geometry(const Geometry &tm0, const Geometry &tm1);
	~Geometry();

	unsigned int vertexCount;
	unsigned int triangleCount;
	// Siggned because of the delta parameter
	uint nLights;
	uint meshLightOffset;


	Point *vertices;
	Normal *vertNormals;
	RGB *vertColors;
	Triangle *triangles;
	TriangleLight *lights;
	char* chunk;


	__HD__
	unsigned int SampleLights(const float u) const {
		// One Uniform light strategy
		const unsigned int lightIndex = Min(Floor2UInt(nLights * u), nLights - 1);

		return lightIndex;
	}
	__HD__
	bool IsLight(const unsigned int index) const {
		return (index >= meshLightOffset);
	}

	void Inline(){
		uint chunk_length = vertexCount*sizeof(Point) +  vertexCount*sizeof(Normal) +
				vertexCount*sizeof(RGB) +  triangleCount*sizeof(Triangle) + nLights*sizeof(TriangleLight);

		chunk = (char*)malloc(chunk_length);

		char* vertices_c = chunk;
		char* vertNormals_c = vertices_c + vertexCount*sizeof(Point);
		char* vertColors_c = vertNormals_c + vertexCount*sizeof(Normal);
		char* triangles_c = vertColors_c + vertexCount*sizeof(RGB);
		char* lights_c = triangles_c + triangleCount*sizeof(Triangle);


		memcpy(vertices_c,vertices, vertexCount*sizeof(Point));
		memcpy(vertNormals_c,vertNormals, vertexCount*sizeof(Normal));
		memcpy(vertColors_c,vertColors, vertexCount*sizeof(RGB));
		memcpy(triangles_c,triangles, triangleCount*sizeof(Triangle));
		memcpy(lights_c,lights, nLights*sizeof(TriangleLight));

		vertices = (Point*)vertices_c;
		vertNormals = (Normal*)vertNormals_c;
		vertColors = (RGB*)vertColors_c;
		triangles = (Triangle*)triangles_c;
		lights = (TriangleLight*)lights_c;

	}

};



#endif	/* _PLYOBJECT_H */

