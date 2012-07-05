//#include <cstdio>
//#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "display/film.h"
#include "display/displayfunc.h"
#include "path.h"
#include "pathtracer.h"
#include "randomgen.h"
#include "geometry.h"
#include "geometry/bvhaccel.h"
#include "camera.h"

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/thread/mutex.hpp>

#include "papitest.h"

void checkCUDAmemory(char* t) {

	//cudaDeviceSynchronize();
	size_t free, total;
	cuMemGetInfo(&free, &total);
	fprintf(stderr,"%s mem %ld total %ld\n", t, free / 1024 / 1024, total / 1024 / 1024);

}

void checkCUDAError(char* t) {

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error %s: %s.\n", t, cudaGetErrorString(err));

		char * tmp = strdup(string("").c_str());
		checkCUDAmemory(tmp);
		free(tmp);

		exit(-1);
	}
}

/**
 * sample origins are pre-generated in host side. Stratified approach.
 */
void preGeneratePaths(prePath* prePaths){


	uint currentSampleScreenX = 0;
	uint currentSampleScreenY = 0;
	uint currentSubSampleIndex = 0;

	for (int _i = 0; _i < PATH_COUNT; _i++) {

		uint scrX, scrY;

		// In order to improve ray coherency
		uint stepX = currentSubSampleIndex % 4;
		uint stepY = currentSubSampleIndex / 4;

		scrX = currentSampleScreenX;
		scrY = currentSampleScreenY;

		currentSubSampleIndex++;
		if (currentSubSampleIndex == SPP) {
		currentSubSampleIndex = 0;
			currentSampleScreenX++;
			if (currentSampleScreenX >= WIDTH) {

				currentSampleScreenX = 0;
				currentSampleScreenY++;

				if (currentSampleScreenY >= HEIGHT) {
					currentSampleScreenY = 0;
				}
			}
		}

		//	const float r1 = (stepX + getFloatRNG(&p->seed)) / 4.f - .5f;
		//	const float r2 = (stepY + getFloatRNG(&p->seed)) / 4.f - .5f;
		//
		//

		const float r1 = (stepX ) / 4.f;
		const float r2 = (stepY ) / 4.f;

		prePaths[_i].screenX = scrX + r1;
		prePaths[_i].screenY = scrY + r2;

		}

}

namespace CPU {

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

RGB Sample_L(uint triIndex, const Geometry *objs, const Point &p, const Normal &N, const float u0,
		const float u1, float *pdf, Ray *shadowRay) {

	const Triangle &tri = objs->triangles[objs->meshLightOffset + triIndex];

	float area = objs->lights[triIndex].area;

	Point samplePoint;
	float b0, b1, b2;

	tri.Sample(objs->vertices, u0, u1, &samplePoint, &b0, &b1, &b2);
	Normal sampleN = objs->vertNormals[tri.v[0]]; // Light sources are supposed to be flat

	Vector wi = samplePoint - p;
	const float distanceSquared = wi.LengthSquared();
	const float distance = sqrtf(distanceSquared);
	wi /= distance;

	float SampleNdotMinusWi = Dot(sampleN, -wi);
	float NdotMinusWi = Dot(N, wi);
	if ((SampleNdotMinusWi <= 0.f) || (NdotMinusWi <= 0.f)) {
		*pdf = 0.f;
		return RGB(0.f, 0.f, 0.f);
	}

	*shadowRay = Ray(p, wi, RAY_EPSILON, distance - RAY_EPSILON);
	*pdf = distanceSquared / (SampleNdotMinusWi * NdotMinusWi * area);

	// Return interpolated color
	return tri.InterpolateColor(objs->vertColors, b0, b1, b2);
}

bool BBox_IntersectP(BBox bbox, const Ray &ray, float *hitt0, float *hitt1) {
	float t0 = ray.mint, t1 = ray.maxt;
	for (int i = 0; i < 3; ++i) {
		// Update interval for _i_th bounding box slab
		float invRayDir = 1.f / ray.d[i];
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

bool Shade(Path* path, Geometry *geometry, BVHAccel *bvh, const RayHit& rayHit) {

	uint tracedShadowRayCount;

	if (rayHit.index == 0xffffffffu) {
		return false;
	}

	// Something was hit
	unsigned int currentTriangleIndex = rayHit.index;
	RGB triInterpCol = geometry->triangles[currentTriangleIndex].InterpolateColor(
			geometry->vertColors, rayHit.b1, rayHit.b2);
	Normal shadeN = geometry->triangles[currentTriangleIndex].InterpolateNormal(
			geometry->vertNormals, rayHit.b1, rayHit.b2);

	// Calculate next step
	path->depth++;

	// Check if I have to stop
	if (path->depth >= MAX_PATH_DEPTH) {
		// Too depth, terminate the path
		return false;
	} else if (path->depth > 2) {

		// Russian Rulette, maximize cos
		const float p = min(1.f, triInterpCol.filter() * AbsDot(shadeN, path->pathRay.d));

		if (p > getFloatRNG(&path->seed))
			path->throughput /= p;
		else {
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
				&lightPdf[tracedShadowRayCount], &shadowRay[tracedShadowRayCount]);

		// Scale light pdf for ONE_UNIFORM strategy
		lightPdf[tracedShadowRayCount] *= lightStrategyPdf;

		// Using 0.1 instead of 0.0 to cut down fireflies
		if (lightPdf[tracedShadowRayCount] > 0.1f)
			tracedShadowRayCount++;
	}

	RayHit* rh = new RayHit[tracedShadowRayCount];

	for (unsigned int i = 0; i < tracedShadowRayCount; ++i)
		Intersect(shadowRay[i],  rh[i], bvh->bvhTree, geometry->triangles, geometry->vertices);

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

void UpdateScreenBuffer(Film* film, uint width, uint height, RGB *pixelsRadiance, float *pixels) {


	const float weight = SPP;

	unsigned int count = width * height;
	for (unsigned int i = 0, j = 0; i < count; ++i) {


		if (weight == 0.f)
			j += 3;
		else {
			const float invWeight = 1.f / weight;

			pixels[j++] = film->Radiance2PixelFloat(pixelsRadiance[i].r * invWeight);
			pixels[j++] = film->Radiance2PixelFloat(pixelsRadiance[i].g * invWeight);
			pixels[j++] = film->Radiance2PixelFloat(pixelsRadiance[i].b * invWeight);
		}
	}
}

void ParseSceneFile(PerspectiveCamera*& camera, Geometry*& geometry,
		string folder) {

	cerr << "Reading scene: " << SCENE_FILE << endl;

	ifstream file;
	file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);
	file.open(SCENE_FILE, ios::in);

	//--------------------------------------------------------------------------
	// Read light position and radius
	//--------------------------------------------------------------------------

	RGB lightGain;
	file >> lightGain.r;
	file >> lightGain.g;
	file >> lightGain.b;

	cerr << "Light gain: (" << lightGain.r << ", " << lightGain.g << ", " << lightGain.b << ")"
			<< endl;

	//--------------------------------------------------------------------------
	// Read camera position and target
	//--------------------------------------------------------------------------

	Point o;
	file >> o.x;
	file >> o.y;
	file >> o.z;

	Point t;
	file >> t.x;
	file >> t.y;
	file >> t.z;

	// Translate backwards 6x
	//	Vector t2 = -6 * Normalize(t - o);
	//	o += t2;
	//	t += t2;

	cerr << "Camera postion: " << o << endl;
	cerr << "Camera target: " << t << endl;

	camera = new PerspectiveCamera(o, t,WIDTH, HEIGHT);

	//--------------------------------------------------------------------------
	// Read objects .ply file
	//--------------------------------------------------------------------------

	string plyFileName;
	file >> plyFileName;

	plyFileName = folder + plyFileName;

	cerr << "PLY objects file name: " << plyFileName << endl;

	Geometry objects(plyFileName);

	//--------------------------------------------------------------------------
	// Read lights .ply file
	//--------------------------------------------------------------------------

	file >> plyFileName;

	plyFileName = folder + plyFileName;

	cerr << "PLY lights file name: " << plyFileName << endl;

	Geometry meshLights(plyFileName);
	// Scale lights intensity
	for (unsigned int i = 0; i < meshLights.vertexCount; ++i)
		meshLights.vertColors[i] *= lightGain;

	//--------------------------------------------------------------------------
	// Join the ply objects
	//--------------------------------------------------------------------------

	geometry = new Geometry(objects, meshLights);
	geometry->meshLightOffset = objects.triangleCount;

	cerr << "Vertex count: " << geometry->vertexCount << " (" << (geometry->vertexCount
			* sizeof(Point) / 1024) << "Kb)" << endl;
	cerr << "Triangle count: " << geometry->triangleCount << " (" << (geometry->triangleCount
			* sizeof(Triangle) / 1024) << "Kb)" << endl;

	//--------------------------------------------------------------------------
	// Create light sources list
	//--------------------------------------------------------------------------

	geometry->nLights = geometry->triangleCount - geometry->meshLightOffset;
	geometry->lights = new TriangleLight[geometry->nLights];
	for (size_t i = 0; i < geometry->nLights; ++i)
		new (&(geometry->lights[i])) TriangleLight(i + geometry->meshLightOffset,
				&(geometry->triangles[i + geometry->meshLightOffset]), geometry->vertices);

}

void DistrurbeSample(Path *p) {

	p->screenX += getFloatRNG(&p->seed) / 4.f;
	p->screenY += getFloatRNG(&p->seed) / 4.f;

}

} // namespace CPU

using namespace CPU;

void kernel_wrapper(PerspectiveCamera camera, RGB *pixelsRadiance,
		 BVHAccelArrayNode* bvh, Point *vertices, Normal *vertNormals,
		RGB *vertColors, Triangle *triangles, TriangleLight *lights, Geometry geometry,
		prePath* paths);

int main(int argc, char *argv[]) {


	cerr << "PARAMETERS:" << WIDTH << "x" << HEIGHT << "x" << SPP << "x" << SHADOWRAY << "x" << MAX_PATH_DEPTH << endl;


	// Entities
	RGB *pixelsRadiance;
	float *pixels;
	PerspectiveCamera *camera = new PerspectiveCamera();
	BVHAccel *bvh;
	Geometry *geometry = new Geometry(); // include object and light areas

	ParseSceneFile(camera, geometry,  "/home/cpd19830/cg/ifr/pi/github/pathtracer13/");

	bvh = new BVHAccel(geometry->triangleCount, geometry->triangles, geometry->vertices);

	pixelsRadiance = new RGB[PIXEL_COUNT];
	pixels = new float[PIXEL_COUNT * 3];

	memset(pixelsRadiance, 0, PIXEL_COUNT* sizeof(RGB));
	memset(pixels, 0, 3 * PIXEL_COUNT * sizeof(float));

	printf("size of prepath %d\n", sizeof(prePath));
	prePath* paths = new prePath[PATH_COUNT];

	preGeneratePaths(paths);


	PERFORM_acc_timer* timer = new PERFORM_acc_timer();

#ifdef CUDA

	CUptiResult cuptiErr;
	CUpti_SubscriberHandle subscriber;
	RuntimeApiTrace_t trace;

	cuptiErr = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) getEventValueCallback, &trace);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiSubscribe");

	cuptiErr = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEnableCallback");

	fprintf(stderr, "\n\n\nSTARTING PAPI\n\n\n");
	papiTestInit(argc-1, argv+1);


	timer->start();

	//// Copy data to GPU

	geometry->Inline(); // inline all the geometry memory into a single memory chunk to ease memory transfer

	RGB *pixelsRadiance_D;
	cudaMalloc((void**) &pixelsRadiance_D, PIXEL_COUNT * sizeof(RGB));
	cudaMemset(pixelsRadiance_D,0, PIXEL_COUNT * sizeof(RGB));

	uint chunk_length = geometry->vertexCount * sizeof(Point) + geometry->vertexCount
	* sizeof(Normal) + geometry->vertexCount * sizeof(RGB) + geometry->triangleCount
	* sizeof(Triangle) + geometry->nLights * sizeof(TriangleLight);

	char* geometry_chunk_D;
	cudaMalloc((void**) &geometry_chunk_D, chunk_length);
	cudaMemcpy(geometry_chunk_D, geometry->chunk, chunk_length, cudaMemcpyHostToDevice);


	// get chunk memory pointers

	char* vertices_c = geometry_chunk_D;
	char * vertNormals_c = vertices_c + geometry->vertexCount * sizeof(Point);
	char * vertColors_c = vertNormals_c + geometry->vertexCount * sizeof(Normal);
	char * triangles_c = vertColors_c + geometry->vertexCount * sizeof(RGB);
	char * lights_c = triangles_c + geometry->triangleCount * sizeof(Triangle);

	Point* vertices_D = (Point*) vertices_c;
	Normal* vertNormals_D = (Normal*) vertNormals_c;
	RGB* vertColors_D = (RGB*) vertColors_c;
	Triangle* triangles_D = (Triangle*) triangles_c;
	TriangleLight* lights_D = (TriangleLight*) lights_c;

	BVHAccelArrayNode* bvh_D;
	cudaMalloc((void**) &bvh_D, bvh->nNodes * sizeof(BVHAccelArrayNode));
	cudaMemcpy(bvh_D, bvh->bvhTree, bvh->nNodes * sizeof(BVHAccelArrayNode), cudaMemcpyHostToDevice);




	prePath* paths_D;
	cudaMalloc((void**) &paths_D, PATH_COUNT * sizeof(prePath));
	cudaMemcpy(paths_D, paths, PATH_COUNT * sizeof(prePath), cudaMemcpyHostToDevice);

	/////

	// call the cuda function
	kernel_wrapper( *camera, pixelsRadiance_D,  bvh_D, vertices_D,
			vertNormals_D, vertColors_D, triangles_D, lights_D, *geometry, paths_D);


	// retrieve output data
	cudaMemcpy(pixelsRadiance, pixelsRadiance_D, PIXEL_COUNT* sizeof(RGB),
			cudaMemcpyDeviceToHost);


	timer->stop();

	{
		char * tmp = strdup("");
	checkCUDAmemory(tmp);
		free(tmp);
	}

#else

	//boost::mutex radianceMutex;

	printf("Rendering process started\n");

	timer->start();

#pragma omp parallel for schedule(dynamic)
	for (uint ipath = 0; ipath < PATH_COUNT ; ipath++) {

		bool not_done = false;

		Path p = Path(&paths[ipath]);

		initRND(&(p.seed), ipath);

		DistrurbeSample(&p);

		camera->GenerateRay(&p, &p.pathRay);

		// Process path
		do {

			Ray rb = p.pathRay;
			RayHit hit;
			//bvh->Intersect(rb, hit);
			Intersect(rb, hit, bvh->bvhTree, geometry->triangles, geometry->vertices);
			not_done = Shade(&p, geometry, bvh, hit);

		} while (not_done);

		int x = (int) p.screenX;
		int y = (int) p.screenY;

		const unsigned int offset = x + y *WIDTH;

		//radianceMutex.lock();

		pixelsRadiance[offset] += p.radiance;

		//radianceMutex.unlock();

//		if (ipath % 100000 == 0) {
//			float perc = (ipath * 100.0f) / PATH_COUNT;
//			printf("Processed %.0f\%\n", perc);
//		}

	}

	timer->stop();

	papiTestCleanup();

#endif

	fprintf(stdout, "Rendering time: %s\n", timer->print());

	film = new Film(WIDTH,HEIGHT);

	UpdateScreenBuffer(film, WIDTH, HEIGHT, pixelsRadiance, pixels);

	film->pixels = pixels;

	InitGlut(argc, argv, WIDTH ,HEIGHT);

	film->SavePPM("image.ppm");

	RunGlut();

	return EXIT_SUCCESS;
}
