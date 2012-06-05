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

#ifndef _FILM_H
#define	_FILM_H

#include <cstddef>
#include <cmath>

#include "utils.h"
#include "geometry/vector.h"
#include "geometry/spectrum.h"

#include "pathtracer.h"

#define GAMMA_TABLE_SIZE 1024
#define FILTER_TABLE_SIZE 16

class Film {
public:

	unsigned int width, height;
	unsigned int pixelCount;

	unsigned int statsTotalSampleCount;
	double statsStartSampleTime, statsAvgSampleSec;

	RGB *pixelsRadiance;
	float *pixelWeights;
	float *pixels;

	float gammaTable[GAMMA_TABLE_SIZE];

	Film(const unsigned int w, unsigned int h) {
		pixelsRadiance = NULL;
		pixelWeights = NULL;
		pixels = NULL;

		InitGammaTable();

		Init(w, h);
	}

	void StartSampleTime() {
		statsStartSampleTime = WallClockTime();
	}

	unsigned int GetWidth() {
		return width;
	}
	unsigned int GetHeight() {
		return height;
	}
	unsigned int GetTotalSampleCount() {
		return statsTotalSampleCount;
	}

	double GetTotalTime() {
		return WallClockTime() - statsStartSampleTime;
	}
	double GetAvgSampleSec() {
		const double elapsedTime = WallClockTime() - statsStartSampleTime;
		const double k = (elapsedTime < 10.0) ? 1.0 : (1.0 / (2.5 * elapsedTime));
		statsAvgSampleSec = k * statsTotalSampleCount / elapsedTime + (1.0 - k) * statsAvgSampleSec;

		return statsAvgSampleSec;
	}

	virtual ~Film() {
		if (pixelsRadiance)
			delete[] pixelsRadiance;
		if (pixelWeights)
			delete[] pixelWeights;
		if (pixels)
			delete[] pixels;
	}

	virtual void Init(const unsigned int w, unsigned int h) {

		if (pixelsRadiance)
			delete[] pixelsRadiance;
		if (pixelWeights)
			delete[] pixelWeights;
		if (pixels)
			delete[] pixels;

		pixelCount = w * h;

		pixelsRadiance = new RGB[pixelCount];
		pixels = new float[pixelCount * 3];
		pixelWeights = new float[pixelCount];

		for (unsigned int i = 0, j = 0; i < pixelCount; ++i) {
			pixelsRadiance[i] = 0.f;
			pixelWeights[i] = 0.f;
			pixels[j++] = 0.f;
			pixels[j++] = 0.f;
			pixels[j++] = 0.f;
		}

		width = w;
		height = h;
		cerr << "Film size " << width << "x" << height << endl;

		statsTotalSampleCount = 0;
		statsAvgSampleSec = 0.0;
		statsStartSampleTime = WallClockTime();
	}

	virtual void Reset() {

		for (unsigned int i = 0; i < pixelCount; ++i) {
			pixelsRadiance[i] = 0.f;
			pixelWeights[i] = 0.f;
		}

		statsTotalSampleCount = 0;
		statsAvgSampleSec = 0.0;
		statsStartSampleTime = WallClockTime();
	}
	__HD__
	void UpdateScreenBuffer() {
		unsigned int count = width * height;
				for (unsigned int i = 0, j = 0; i < count; ++i) {
					const float weight = pixelWeights[i];

					if (weight == 0.f)
						j += 3;
					else {
						const float invWeight = 1.f / weight;

						pixels[j++] = Radiance2PixelFloat(pixelsRadiance[i].r * invWeight);
						pixels[j++] = Radiance2PixelFloat(pixelsRadiance[i].g * invWeight);
						pixels[j++] = Radiance2PixelFloat(pixelsRadiance[i].b * invWeight);
					}
				}
	}

	const float *GetScreenBuffer() const {
		return pixels;
	}


	__HD__
	void SavePPM(const string &fileName) {

		// Update pixels
		UpdateScreenBuffer();

		const float *pixels = GetScreenBuffer();

		ofstream file;
		file.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);
		file.open(fileName.c_str(), ios::out);

		file << "P3\n" << width << " " << height << "\n255\n";

		for (unsigned int y = 0; y < height; ++y) {
			for (unsigned int x = 0; x < width; ++x) {
				const int offset = 3 * (x + (height - y - 1) * width);
				const int r = (int) (pixels[offset] * 255.f + .5f);
				const int g = (int) (pixels[offset + 1] * 255.f + .5f);
				const int b = (int) (pixels[offset + 2] * 255.f + .5f);

				file << r << " " << g << " " << b << " ";
			}
		}

		file.close();
	}

	void InitGammaTable() {
		float x = 0.f;
		const float dx = 1.f / GAMMA_TABLE_SIZE;
		for (int i = 0; i < GAMMA_TABLE_SIZE; ++i, x += dx)
			gammaTable[i] = powf(Clamp(x, 0.f, 1.f), 1.f / 2.2f);
	}
	__HD__
	float Radiance2PixelFloat(const float x) const {
		// Very slow !
		//return powf(Clamp(x, 0.f, 1.f), 1.f / 2.2f);

		const unsigned int index = Min<unsigned int> (
				Floor2UInt(GAMMA_TABLE_SIZE * Clamp(x, 0.f, 1.f)), GAMMA_TABLE_SIZE - 1);
		return gammaTable[index];
	}




};

#endif	/* _FILM_H */
