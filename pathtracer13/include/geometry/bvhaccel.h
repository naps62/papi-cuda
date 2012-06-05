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
 *   Lux Renderer website : http://www.luxrender.org                       *
 ***************************************************************************/

#ifndef _BVHACCEL_H
#define	_BVHACCEL_H

#include <vector>

#include "vector.h"
#include "point.h"
#include "triangle.h"

using namespace std;

struct BVHAccelTreeNode {
	BBox bbox;
	unsigned int primitive;
	BVHAccelTreeNode *leftChild;
	BVHAccelTreeNode *rightSibling;
};

struct BVHAccelArrayNode {
	BBox bbox;
	unsigned int primitive;
	unsigned int skipIndex;
};

// BVHAccel Declarations
class  BVHAccel{
public:
	// BVHAccel Public Methods
	BVHAccel(const unsigned int triangleCount, const Triangle *p, const Point *v);
	~BVHAccel();

//	bool Intersect(const Ray &ray, RayHit *hit) const;

	unsigned int nPrims;
	const Point *vertices;
	const Triangle *triangles;
	unsigned int nNodes;
	BVHAccelArrayNode *bvhTree;

	// BVHAccel Private Methods
	BVHAccelTreeNode *BuildHierarchy(vector<BVHAccelTreeNode *> &list, unsigned int begin, unsigned int end, unsigned int axis);
	void FindBestSplit(vector<BVHAccelTreeNode *> &list, unsigned int begin, unsigned int end, float *splitValue, unsigned int *bestAxis);
	unsigned int BuildArray(BVHAccelTreeNode *node, unsigned int offset);
	void FreeHierarchy(BVHAccelTreeNode *node);

};

#endif	/* _BVHACCEL_H */
