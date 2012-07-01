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

// Boundary Volume Hierarchy accelerator
// Based of "Efficiency Issues for Ray Tracing" by Brian Smits
// Available at http://www.cs.utah.edu/~bes/papers/fastRT/paper.html

#include <iostream>
#include <functional>
#include <algorithm>
#include <limits>

#include "geometry/bvhaccel.h"

using std::bind2nd;
using std::ptr_fun;

using namespace std;

// BVHAccel Method Definitions

BVHAccel::BVHAccel(const unsigned triangleCount, const Triangle *p, const Point *v):
		nPrims(triangleCount), vertices(v), triangles(p) {
//	// Make sure treeType is 2, 4 or 8
//	if (treetype <= 2) treeType = 2;
//	else if (treetype <= 4) treeType = 4;
//	else treeType = 8;

	vector<BVHAccelTreeNode *> bvList;
	for (unsigned int i = 0; i < nPrims; ++i) {
		BVHAccelTreeNode *ptr = new BVHAccelTreeNode();
		ptr->bbox = p[i].WorldBound(v);
		// NOTE - Ratow - Expand bbox a little to make sure rays collide
		ptr->bbox.Expand(RAY_EPSILON);
		ptr->primitive = i;
		ptr->leftChild = NULL;
		ptr->rightSibling = NULL;
		bvList.push_back(ptr);
	}

	cerr << "Building Bounding Volume Hierarchy, primitives: " << nPrims << endl;

	nNodes = 0;
	BVHAccelTreeNode *rootNode = BuildHierarchy(bvList, 0, bvList.size(), 2);

	cerr << "Pre-processing Bounding Volume Hierarchy, total nodes: " << nNodes << endl;

	bvhTree = new BVHAccelArrayNode[nNodes];
	BuildArray(rootNode, 0);
	FreeHierarchy(rootNode);

	cerr << "Finished building Bounding Volume Hierarchy array" << endl;
}

BVHAccel::~BVHAccel() {
	delete bvhTree;
}

void BVHAccel::FreeHierarchy(BVHAccelTreeNode *node) {
	if (node) {
		FreeHierarchy(node->leftChild);
		FreeHierarchy(node->rightSibling);

		delete node;
	}
}

// Build an array of comparators for each axis

bool bvh_ltf_x(BVHAccelTreeNode *n, float v) {
	return n->bbox.pMax.x + n->bbox.pMin.x < v;
}

bool bvh_ltf_y(BVHAccelTreeNode *n, float v) {
	return n->bbox.pMax.y + n->bbox.pMin.y < v;
}

bool bvh_ltf_z(BVHAccelTreeNode *n, float v) {
	return n->bbox.pMax.z + n->bbox.pMin.z < v;
}

bool (* const bvh_ltf[3])(BVHAccelTreeNode *n, float v) = {bvh_ltf_x, bvh_ltf_y, bvh_ltf_z};

BVHAccelTreeNode *BVHAccel::BuildHierarchy(vector<BVHAccelTreeNode *> &list, unsigned int begin, unsigned int end, unsigned int axis) {


	const int treeType = 4; // Tree type to generate (2 = binary, 4 = quad, 8 = octree)


	unsigned int splitAxis = axis;
	float splitValue;

	nNodes += 1;
	if (end - begin == 1) // Only a single item in list so return it
		return list[begin];

	BVHAccelTreeNode *parent = new BVHAccelTreeNode();
	parent->primitive = 0xffffffffu;
	parent->leftChild = NULL;
	parent->rightSibling = NULL;

	vector<unsigned int> splits;
	splits.reserve(treeType + 1);
	splits.push_back(begin);
	splits.push_back(end);
	for (unsigned int i = 2; i <= treeType; i *= 2) { // Calculate splits, according to tree type and do partition
		for (unsigned int j = 0, offset = 0; j + offset < i && splits.size() > j + 1; j += 2) {
			if (splits[j + 1] - splits[j] < 2) {
				j--;
				offset++;
				continue; // Less than two elements: no need to split
			}

			FindBestSplit(list, splits[j], splits[j + 1], &splitValue, &splitAxis);

			vector<BVHAccelTreeNode *>::iterator it =
					partition(list.begin() + splits[j], list.begin() + splits[j + 1], bind2nd(ptr_fun(bvh_ltf[splitAxis]), splitValue));
			unsigned int middle = distance(list.begin(), it);
			middle = max(splits[j] + 1, min(splits[j + 1] - 1, middle)); // Make sure coincidental BBs are still split
			splits.insert(splits.begin() + j + 1, middle);
		}
	}

	BVHAccelTreeNode *child, *lastChild;
	// Left Child
	child = BuildHierarchy(list, splits[0], splits[1], splitAxis);
	parent->leftChild = child;
	parent->bbox = child->bbox;
	lastChild = child;

	// Add remaining children
	for (unsigned int i = 1; i < splits.size() - 1; i++) {
		child = BuildHierarchy(list, splits[i], splits[i + 1], splitAxis);
		lastChild->rightSibling = child;
		parent->bbox = Union(parent->bbox, child->bbox);
		lastChild = child;
	}

	return parent;
}

void BVHAccel::FindBestSplit(vector<BVHAccelTreeNode *> &list, unsigned int begin, unsigned int end, float *splitValue, unsigned int *bestAxis) {

	const int costSamples = 0; // Samples to get for cost minimization
	const int isectCost = 80;
	const int traversalCost = 10;
	const float emptyBonus = 0.5f;


	if (end - begin == 2) {
		// Trivial case with two elements
		*splitValue = (list[begin]->bbox.pMax[0] + list[begin]->bbox.pMin[0] +
				list[end - 1]->bbox.pMax[0] + list[end - 1]->bbox.pMin[0]) / 2;
		*bestAxis = 0;
	} else {
		// Calculate BBs mean center (times 2)
		Point mean2(0, 0, 0), var(0, 0, 0);
		for (unsigned int i = begin; i < end; i++)
			mean2 += list[i]->bbox.pMax + list[i]->bbox.pMin;
		mean2 /= end - begin;

		// Calculate variance
		for (unsigned int i = begin; i < end; i++) {
			Vector v = list[i]->bbox.pMax + list[i]->bbox.pMin - mean2;
			v.x *= v.x;
			v.y *= v.y;
			v.z *= v.z;
			var += v;
		}
		// Select axis with more variance
		if (var.x > var.y && var.x > var.z)
			*bestAxis = 0;
		else if (var.y > var.z)
			*bestAxis = 1;
		else
			*bestAxis = 2;

		if (costSamples > 1) {
			BBox nodeBounds;
			for (unsigned int i = begin; i < end; i++)
				nodeBounds = Union(nodeBounds, list[i]->bbox);

			Vector d = nodeBounds.pMax - nodeBounds.pMin;
			float totalSA = (2.f * (d.x * d.y + d.x * d.z + d.y * d.z));
			float invTotalSA = 1.f / totalSA;

			// Sample cost for split at some points
			float increment = 2 * d[*bestAxis] / (costSamples + 1);
			float bestCost = INFINITY;
			for (float splitVal = 2 * nodeBounds.pMin[*bestAxis] + increment; splitVal < 2 * nodeBounds.pMax[*bestAxis]; splitVal += increment) {
				int nBelow = 0, nAbove = 0;
				BBox bbBelow, bbAbove;
				for (unsigned int j = begin; j < end; j++) {
					if ((list[j]->bbox.pMax[*bestAxis] + list[j]->bbox.pMin[*bestAxis]) < splitVal) {
						nBelow++;
						bbBelow = Union(bbBelow, list[j]->bbox);
					} else {
						nAbove++;
						bbAbove = Union(bbAbove, list[j]->bbox);
					}
				}
				Vector dBelow = bbBelow.pMax - bbBelow.pMin;
				Vector dAbove = bbAbove.pMax - bbAbove.pMin;
				float belowSA = 2 * ((dBelow.x * dBelow.y + dBelow.x * dBelow.z + dBelow.y * dBelow.z));
				float aboveSA = 2 * ((dAbove.x * dAbove.y + dAbove.x * dAbove.z + dAbove.y * dAbove.z));
				float pBelow = belowSA * invTotalSA;
				float pAbove = aboveSA * invTotalSA;
				float eb = (nAbove == 0 || nBelow == 0) ? emptyBonus : 0.f;
				float cost = traversalCost + isectCost * (1.f - eb) * (pBelow * nBelow + pAbove * nAbove);
				// Update best split if this is lowest cost so far
				if (cost < bestCost) {
					bestCost = cost;
					*splitValue = splitVal;
				}
			}
		} else {
			// Split in half around the mean center
			*splitValue = mean2[*bestAxis];
		}
	}
}

unsigned int BVHAccel::BuildArray(BVHAccelTreeNode *node, unsigned int offset) {
	// Build array by recursively traversing the tree depth-first
	while (node) {
		BVHAccelArrayNode *p = &bvhTree[offset];

		p->bbox = node->bbox;
		p->primitive = node->primitive;
		offset = BuildArray(node->leftChild, offset + 1);
		p->skipIndex = offset;

		node = node->rightSibling;
	}

	return offset;
}


//bool BVHAccel::Intersect(const Ray &ray, RayHit *rayHit) const {
//
//	rayHit->t = INFINITY;
//	rayHit->index = 0xffffffffu;
//	unsigned int currentNode = 0; // Root Node
//	unsigned int stopNode = bvhTree[0].skipIndex; // Non-existent
//	bool hit = false;
//	RayHit triangleHit;
//
//
//	while (currentNode < stopNode) {
//		if (bvhTree[currentNode].bbox.IntersectP(ray)) {
//			if (bvhTree[currentNode].primitive != 0xffffffffu) {
//				//float tt, b1, b2;
//				if (triangles[bvhTree[currentNode].primitive].Intersect(ray, vertices, &triangleHit)) {
//					hit = true; // Continue testing for closer intersections
//					if (triangleHit.t < rayHit->t) {
//						rayHit->t = triangleHit.t;
//						rayHit->b1 = triangleHit.b1;
//						rayHit->b2 = triangleHit.b2;
//						rayHit->index = bvhTree[currentNode].primitive;
//					}
//				}
//			}
//
//			currentNode++;
//		} else
//			currentNode = bvhTree[currentNode].skipIndex;
//	}
//
//	return hit;
//}