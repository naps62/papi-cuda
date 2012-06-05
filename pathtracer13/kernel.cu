
#include "pathtracer.h"
#include "path.h"
#include "camera.h"
#include "ray.h"
#include "geometry/bvhaccel.h"
#include "geometry/triangle.h"
#include "geometry/light.h"
#include "geometry.h"
#include "math.h"
#include "utils.h"
#include "randomgen.h"
#include "config.h"



__device__
bool TriangleIntersect(Triangle* tri, const Ray &ray, const Point *verts, RayHit *rayHit) {

		const Point &p0 = verts[tri->v[0]];
		const Point &p1 = verts[tri->v[1]];
		const Point &p2 = verts[tri->v[2]];



		const Vector e1 = p1 - p0;
		const Vector e2 = p2 - p0;
		const Vector s1 = Cross(ray.d, e2);

		const float divisor = Dot(s1, e1);
		if (divisor == 0.f)
			return false;

		const float invDivisor = 1.f / divisor;

		// Compute first barycentric coordinate
		const Vector d = ray.o - p0;
		const float b1 = Dot(d, s1) * invDivisor;
		if (b1 < 0.f)
			return false;

		// Compute second barycentric coordinate
		const Vector s2 = Cross(d, e1);
		const float b2 = Dot(ray.d, s2) * invDivisor;
		if (b2 < 0.f)
			return false;

		const float b0 = 1.f - b1 - b2;
		if (b0 < 0.f)
			return false;

		// Compute _t_ to intersection point
		const float t = Dot(e2, s2) * invDivisor;
		if (t < ray.mint || t > ray.maxt)
			return false;

		rayHit->t = t;
		rayHit->b1 = b1;
		rayHit->b2 = b2;

		return true;
	}

__device__
bool BBox_IntersectP(BBox bbox, const Ray &ray, float *hitt0, float *hitt1) {

	float t0 = ray.mint, t1 = ray.maxt;
	for (int i = 0; i < 3; ++i) {
		// Update interval for _i_th bounding box slab
		const float invRayDir = 1.f / ray.d[i];
		float tNear = (bbox.pMin[i] - ray.o[i]) * invRayDir;
		float tFar = (bbox.pMax[i] - ray.o[i]) * invRayDir;
		// Update parametric interval from slab intersection $t$s
		if (tNear > tFar)
			swap(&tNear, &tFar);
		t0 = tNear > t0 ? tNear : t0;
		t1 = tFar < t1 ? tFar : t1;
		if (t0 > t1)
			return false;
	}
	if (hitt0)
		*hitt0 = t0;
	if (hitt1)
		*hitt1 = t1;
	return true;
}

__device__
bool Intersect(const Ray &ray, RayHit& rayHit, BVHAccelArrayNode* bvhTree, Triangle *triangles,
		Point *vertices) {

	rayHit.t = INFINITY;
	rayHit.index = 0xffffffffu;
	unsigned int currentNode = 0; // Root Node
	unsigned int stopNode = bvhTree[0].skipIndex; // Non-existent
	bool hit = false;
	RayHit triangleHit;

	while (currentNode < stopNode) {
		if (BBox_IntersectP(bvhTree[currentNode].bbox, ray, NULL, NULL)) {
			if (bvhTree[currentNode].primitive != 0xffffffffu) {
				//float tt, b1, b2;
				Triangle t = triangles[bvhTree[currentNode].primitive];
				if (TriangleIntersect(&t,ray, vertices, &triangleHit)) {
					hit = true; // Continue testing for closer intersections
					if (triangleHit.t < rayHit.t) {
						rayHit.t = triangleHit.t;
						rayHit.b1 = triangleHit.b1;
						rayHit.b2 = triangleHit.b2;
						rayHit.index = bvhTree[currentNode].primitive;
					}
				}
			}

			currentNode++;
		} else
			currentNode = bvhTree[currentNode].skipIndex;
	}

	return hit;
}

/**
 * Light sampling based on distance and cos
 */
__device__
RGB Sample_L(uint triIndex, Geometry *objs, const Point &p, const Normal &N, const float u0,
		const float u1, float *pdf, Ray *shadowRay,Point *vertices,
		Normal *vertNormals, RGB *vertColors, Triangle *triangles, TriangleLight *lights) {

	const Triangle &tri = triangles[objs->meshLightOffset + triIndex];

	const float area = lights[triIndex].area;

	Point samplePoint;
	float b0, b1, b2;

	tri.Sample(vertices, u0, u1, &samplePoint, &b0, &b1, &b2);
	Normal sampleN = vertNormals[tri.v[0]]; // Light sources are supposed to be flat

	Vector wi = samplePoint - p;
	const float distanceSquared = wi.LengthSquared();
	const float distance = sqrtf(distanceSquared);
	wi /= distance;

	const float SampleNdotMinusWi = Dot(sampleN, -wi);
	const float NdotMinusWi = Dot(N, wi);
	if ((SampleNdotMinusWi <= 0.f) || (NdotMinusWi <= 0.f)) {
		*pdf = 0.f;
		return RGB(0.f, 0.f, 0.f);
	}

	*shadowRay = Ray(p, wi, RAY_EPSILON, distance - RAY_EPSILON);
	*pdf = distanceSquared / (SampleNdotMinusWi * NdotMinusWi * area);

	// Return interpolated color
	return tri.InterpolateColor(vertColors, b0, b1, b2);
}

__device__
bool Shade(Path* path, Point *vertices, Normal *vertNormals, RGB *vertColors, Triangle *triangles,
		TriangleLight *lights, BVHAccelArrayNode *bvh, RayHit& rayHit, Geometry* geometry) {

	uint tracedShadowRayCount;

	if (rayHit.index == 0xffffffffu) {
		return false;
	}

	// Something was hit
	unsigned int currentTriangleIndex = rayHit.index;
	RGB triInterpCol = triangles[currentTriangleIndex].InterpolateColor(vertColors, rayHit.b1,
			rayHit.b2);
	Normal shadeN = triangles[currentTriangleIndex].InterpolateNormal(vertNormals, rayHit.b1,
			rayHit.b2);


	// Calculate next step
	path->depth++;

	// Check if I have to stop
	if (path->depth >= MAX_PATH_DEPTH) {
		// Too depth, terminate the path
		return false;
	} else if (path->depth > 2) {

		// Russian Rulette
		const float p = Min(1.f, triInterpCol.filter() * AbsDot(shadeN, path->pathRay.d));

		if (p > getFloatRNG(&path->seed)){
			path->throughput /= p;
		}else {
			// Terminate the path
			return false;
		}
	}

	//--------------------------------------------------------------------------
	// Build the shadow ray
	//--------------------------------------------------------------------------

	// Check if it is a light source
	float RdotShadeN = Dot(path->pathRay.d, shadeN);
	if (geometry->IsLight(currentTriangleIndex)) {
		// Check if we are on the right side of the light source
		if ((path->depth == 1) && (RdotShadeN < 0.f))
			path->radiance += triInterpCol * path->throughput;

		// Terminate the path
		return false;
	}

	if (RdotShadeN > 0.f) {
		// Flip shade  normal
		shadeN = -shadeN;
	} else
		RdotShadeN = -RdotShadeN;

	path->throughput *= RdotShadeN * triInterpCol;

	// Trace shadow rays
	const Point hitPoint = path->pathRay(rayHit.t);

	tracedShadowRayCount = 0;
	const float lightStrategyPdf = static_cast<float> (SHADOWRAY)
			/ static_cast<float> (geometry->nLights);

	float lightPdf[SHADOWRAY];
	RGB lightColor[SHADOWRAY];
	Ray shadowRay[SHADOWRAY];

	for (unsigned int i = 0; i < SHADOWRAY; ++i) {
		// Select the light to sample
		const unsigned int currentLightIndex = geometry->SampleLights(getFloatRNG(&path->seed));
		//	const TriangleLight &light = scene->lights[currentLightIndex];

		// Select a point on the surface
		lightColor[tracedShadowRayCount] = Sample_L(currentLightIndex, geometry, hitPoint, shadeN,
				getFloatRNG(&path->seed), getFloatRNG(&path->seed),
				&lightPdf[tracedShadowRayCount], &shadowRay[tracedShadowRayCount],
				vertices, vertNormals, vertColors, triangles, lights);

		// Scale light pdf for ONE_UNIFORM strategy
		lightPdf[tracedShadowRayCount] *= lightStrategyPdf;

		// Using 0.1 instead of 0.0 to cut down fireflies
		if (lightPdf[tracedShadowRayCount] > 0.1f)
			tracedShadowRayCount++;
	}

	RayHit rh[SHADOWRAY];

	for (unsigned int i = 0; i < tracedShadowRayCount; ++i)
		Intersect(shadowRay[i], rh[i], bvh, triangles, vertices);

	if ((tracedShadowRayCount > 0)) {
		for (unsigned int i = 0; i < tracedShadowRayCount; ++i) {
			const RayHit *shadowRayHit = &rh[i];
			if (shadowRayHit->index == 0xffffffffu) {
				// Nothing was hit, light is visible
				path->radiance += path->throughput * lightColor[i] / lightPdf[i];
			}
		}
	}

	//--------------------------------------------------------------------------
	// Build the next vertex path ray
	//--------------------------------------------------------------------------

	// Calculate exit direction

	float r1 = 2.f * M_PI * getFloatRNG(&path->seed);
	float r2 = getFloatRNG(&path->seed);
	float r2s = sqrt(r2);
	const Vector w(shadeN);

	Vector u;
	if (fabsf(shadeN.x) > .1f) {
		const Vector a(0.f, 1.f, 0.f);
		u = Cross(a, w);
	} else {
		const Vector a(1.f, 0.f, 0.f);
		u = Cross(a, w);
	}
	u = Normalize(u);

	Vector v = Cross(w, u);

	Vector newDir = u * (cosf(r1) * r2s) + v * (sinf(r1) * r2s) + w * sqrtf(1.f - r2);
	newDir = Normalize(newDir);

	path->pathRay.o = hitPoint;
	path->pathRay.d = newDir;

	return true;
}


/**
 * Samples/paths are pre-generated in the host and randomly repositioned inside the stratified pixel area
 */
__device__
void DistrurbeSample(Path *p) {

	p->screenX += getFloatRNG(&p->seed) / 4.f;
	p->screenY += getFloatRNG(&p->seed) / 4.f;

}

/**
 * A cuda thread processes the entire sample path
 */
__global__ void kernel( PerspectiveCamera camera, RGB *pixelsRadiance,
		 BVHAccelArrayNode* bvh, Point *vertices, Normal *vertNormals,
		RGB *vertColors, Triangle *triangles, TriangleLight *lights, Geometry geometry,
		prePath* paths) {

	//const int tID = blockIdx.x * blockDim.x + threadIdx.x;


	const int tID = (gridDim.x * blockDim.x * blockIdx.y * blockDim.y) + (gridDim.x * blockDim.x
			* threadIdx.y) + (blockDim.x * blockIdx.x + threadIdx.x);

	if (tID == 0 )	printf("CUDA rendering process started\n");

	if (tID < PATH_COUNT) {

		bool not_done = false;

		Path p = Path(&paths[tID]);

		initRND(&(p.seed), tID);

		RayHit hit;

		DistrurbeSample(&p);

		camera.GenerateRay(&p, &p.pathRay);

		// Process path
		do {

			Intersect(p.pathRay, hit, bvh, triangles, vertices);
			not_done = Shade(&p, vertices, vertNormals, vertColors, triangles, lights, bvh, hit, &geometry);

		} while (not_done);


		int x = (int) p.screenX;
		int y = (int) p.screenY;

		uint offset = x + y * WIDTH;

		atomicAdd(&(pixelsRadiance[offset].r), p.radiance.r);
		atomicAdd(&(pixelsRadiance[offset].g), p.radiance.g);
		atomicAdd(&(pixelsRadiance[offset].b), p.radiance.b);


	}

}

/**
 * Used to link the host code with cuda code. All the structures can be merged or explicitly copied
 * to device memory to avoid large parameter transfer
 */
void kernel_wrapper(PerspectiveCamera camera, RGB *pixelsRadiance,
		 BVHAccelArrayNode* bvh, Point *vertices, Normal *vertNormals,
		RGB *vertColors, Triangle *triangles, TriangleLight *lights, Geometry geometry,
		prePath* paths) {

	int sqrtn = sqrt(PATH_COUNT);
	dim3 blockDIM = dim3(16, 16);
	dim3 gridDIM = dim3((sqrtn / blockDIM.x) + 1, (sqrtn / blockDIM.y) + 1);

kernel<<<gridDIM,blockDIM >>>( camera, pixelsRadiance,  bvh, vertices, vertNormals,
		vertColors, triangles, lights, geometry,paths);
}
