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

#include <stdexcept>
#include <sstream>
#include <iostream>

#include "geometry.h"

// rply vertex callback
static int VertexCB(p_ply_argument argument) {
	long userIndex = 0;
	void *userData = NULL;
	ply_get_argument_user_data(argument, &userData, &userIndex);

	Point* p = *static_cast<Point **> (userData);

	long vertIndex;
	ply_get_argument_element(argument, NULL, &vertIndex);

	if (userIndex == 0)
		p[vertIndex].x =
			static_cast<float>(ply_get_argument_value(argument));
	else if (userIndex == 1)
		p[vertIndex].y =
			static_cast<float>(ply_get_argument_value(argument));
	else if (userIndex == 2)
		p[vertIndex].z =
			static_cast<float>(ply_get_argument_value(argument));

	return 1;
}

// rply normal callback
static int NormalCB(p_ply_argument argument) {
	long userIndex = 0;
	void *userData = NULL;

	ply_get_argument_user_data(argument, &userData, &userIndex);

	Normal* n = *static_cast<Normal **> (userData);

	long normIndex;
	ply_get_argument_element(argument, NULL, &normIndex);

	if (userIndex == 0)
		n[normIndex].x =
			static_cast<float>(ply_get_argument_value(argument));
	else if (userIndex == 1)
		n[normIndex].y =
			static_cast<float>(ply_get_argument_value(argument));
	else if (userIndex == 2)
		n[normIndex].z =
			static_cast<float>(ply_get_argument_value(argument));

	return 1;
}
// rply color callback
static int ColorCB(p_ply_argument argument) {
	long userIndex = 0;
	void *userData = NULL;
	ply_get_argument_user_data(argument, &userData, &userIndex);

	RGB* c = *static_cast<RGB **> (userData);

	long colIndex;
	ply_get_argument_element(argument, NULL, &colIndex);

	if (userIndex == 0)
		c[colIndex].r =
			static_cast<float>(ply_get_argument_value(argument) / 255.0);
	else if (userIndex == 1)
		c[colIndex].g =
			static_cast<float>(ply_get_argument_value(argument) / 255.0);
	else if (userIndex == 2)
		c[colIndex].b =
			static_cast<float>(ply_get_argument_value(argument) / 255.0);

	return 1;
}

// rply face callback
static int FaceCB(p_ply_argument argument) {
	void *userData = NULL;
	ply_get_argument_user_data(argument, &userData, NULL);

	Triangle *verts = *static_cast<Triangle **> (userData);

	long triIndex;
	ply_get_argument_element(argument, NULL, &triIndex);

	long length, valueIndex;
	ply_get_argument_property(argument, NULL, &length, &valueIndex);

	if (valueIndex >= 0 && valueIndex < 3) {
		verts[triIndex].v[valueIndex] =
				static_cast<unsigned int> (ply_get_argument_value(argument));
	}

	return 1;
}



Geometry::Geometry() {

}

Geometry::Geometry(const string &fileName) {
	p_ply plyfile = ply_open(fileName.c_str(), NULL);
	if (!plyfile) {
		stringstream ss;
		ss << "Unable to read PLY mesh file '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	if (!ply_read_header(plyfile)) {
		stringstream ss;
		ss << "Unable to read PLY header from '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	Point *p;
	long plyNbVerts = ply_set_read_cb(plyfile, "vertex", "x", VertexCB, &p, 0);
	ply_set_read_cb(plyfile, "vertex", "y", VertexCB, &p, 1);
	ply_set_read_cb(plyfile, "vertex", "z", VertexCB, &p, 2);
	if (plyNbVerts <= 0) {
		stringstream ss;
		ss << "No vertices found in '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	Triangle *vi;
	long plyNbTris = ply_set_read_cb(plyfile, "face", "vertex_indices", FaceCB, &vi, 0);
	if (plyNbTris <= 0) {
		stringstream ss;
		ss << "No triangles found in '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	Normal *n;
	long plyNbNormals = ply_set_read_cb(plyfile, "vertex", "nx", NormalCB, &n, 0);
	ply_set_read_cb(plyfile, "vertex", "ny", NormalCB, &n, 1);
	ply_set_read_cb(plyfile, "vertex", "nz", NormalCB, &n, 2);
	if ((plyNbNormals > 0) && (plyNbNormals != plyNbVerts)) {
		stringstream ss;
		ss << "Wrong count of normals in '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	RGB *c;
	long plyNbColors = ply_set_read_cb(plyfile, "vertex", "red", ColorCB, &c, 0);
	ply_set_read_cb(plyfile, "vertex", "green", ColorCB, &c, 1);
	ply_set_read_cb(plyfile, "vertex", "blue", ColorCB, &c, 2);

	if ((plyNbColors <= 0) || ((plyNbColors > 0) && (plyNbColors != plyNbVerts))) {
		stringstream ss;
		ss << "Wrong count of colors in '" << fileName << "'";
		throw runtime_error(ss.str());
	}

	p = new Point[plyNbVerts];
	vi = new Triangle[plyNbTris];
	c = new RGB[plyNbColors];
	n = new Normal[plyNbVerts];

	if (!ply_read(plyfile)) {
		stringstream ss;
		ss << "Unable to parse PLY file '" << fileName << "'";

		delete[] p;
		delete[] vi;
		delete[] c;
		delete[] n;

		throw runtime_error(ss.str());
	}

	ply_close(plyfile);

	vertexCount = plyNbVerts;
	triangleCount = plyNbTris;
	vertices = p;
	triangles = vi;
	vertNormals = n;
	vertColors = c;

	// Scale vertex colors
	for (unsigned int i = 0; i < vertexCount; ++i)
		c[i] *= 0.75f;

	// It looks like normals exported by Blender are bugged
	for (unsigned int i = 0; i < vertexCount; ++i)
		vertNormals[i] = Normal(0.f, 0.f, 0.f);
	for (unsigned int i = 0; i < triangleCount; ++i) {
		const Vector e1 = vertices[triangles[i].v[1]] - vertices[triangles[i].v[0]];
		const Vector e2 = vertices[triangles[i].v[2]] - vertices[triangles[i].v[0]];
		const Normal n = Normal(Normalize(Cross(e1, e2)));
		vertNormals[triangles[i].v[0]] += n;
		vertNormals[triangles[i].v[1]] += n;
		vertNormals[triangles[i].v[2]] += n;
	}
	int printedWarning = 0;
	for (unsigned int i = 0; i < vertexCount; ++i) {
		vertNormals[i] = Normalize(vertNormals[i]);
		// Check for degenerate triangles/normals, they can freeze the GPU
		if (isnan(vertNormals[i].x) || isnan(vertNormals[i].y) || isnan(vertNormals[i].z)) {
			if (printedWarning < 15) {
				cerr << "The model contains a degenerate normal (index " << i << ")" << endl;
				++printedWarning;
			} else if (printedWarning == 15) {
				cerr << "The model contains more degenerate normals" << endl;
				++printedWarning;
			}
			vertNormals[i] = Normal(0.f, 0.f, 1.f);
		}
	}

	/*if (plyNbNormals <= 0) {
		// Calculate normals
		for (unsigned int i = 0; i < triangleCount; ++i) {
			const Vector e1 = vertices[triangles[i].v[1]] - vertices[triangles[i].v[0]];
			const Vector e2 = vertices[triangles[i].v[2]] - vertices[triangles[i].v[0]];
			const Normal n = Normal(Normalize(Cross(e1, e2)));
			vertNormals[triangles[i].v[0]] = n;
			vertNormals[triangles[i].v[1]] = n;
			vertNormals[triangles[i].v[2]] = n;
		}
	}*/
}

Geometry::~Geometry() {

}

Geometry::Geometry(const Geometry &obj0, const Geometry &obj1) {
	vertexCount = obj0.vertexCount + obj1.vertexCount;
	triangleCount = obj0.triangleCount + obj1.triangleCount;

	vertices = new Point[vertexCount];
	triangles = new Triangle[triangleCount];
	vertColors = new RGB[vertexCount];
	vertNormals = new Normal[vertexCount];

	for (unsigned int i = 0; i < obj0.vertexCount; ++i) {
		vertices[i] = obj0.vertices[i];
		vertNormals[i] = obj0.vertNormals[i];
		vertColors[i] = obj0.vertColors[i];
	}
	for (unsigned int i = 0; i < obj1.vertexCount; ++i) {
		vertices[obj0.vertexCount + i] = obj1.vertices[i];
		vertNormals[obj0.vertexCount + i] = obj1.vertNormals[i];
		vertColors[obj0.vertexCount + i] = obj1.vertColors[i];
	}

	for (unsigned int i = 0; i < obj0.triangleCount; ++i)
		triangles[i] = obj0.triangles[i];

	for (unsigned int i = 0; i < obj1.triangleCount; ++i) {
		triangles[obj0.triangleCount + i].v[0] = obj1.triangles[i].v[0] + obj0.vertexCount;
		triangles[obj0.triangleCount + i].v[1] = obj1.triangles[i].v[1] + obj0.vertexCount;
		triangles[obj0.triangleCount + i].v[2] = obj1.triangles[i].v[2] + obj0.vertexCount;
	}
}
