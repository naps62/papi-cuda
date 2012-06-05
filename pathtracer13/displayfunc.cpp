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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#include "display/displayfunc.h"
#include "display/film.h"

Film* film;


static int printHelp = 1;

static void PrintString(void *font, const char *string) {
	int len, i;

	len = (int)strlen(string);
	for (i = 0; i < len; i++)
		glutBitmapCharacter(font, string[i]);
}


static void PrintCaptions() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(0.f, 0.f, 0.f, 0.8f);
	glRecti(0, film->GetHeight() - 15,
			film->GetWidth() - 1, film->GetHeight() - 1);
	glRecti(0, 0, film->GetWidth() - 1, 18);
	glDisable(GL_BLEND);

	// Caption line 0
	glColor3f(1.f, 1.f, 1.f);
	glRasterPos2i(4, 5);

	// Title
	glRasterPos2i(4, film->GetHeight() - 10);

}

void displayFunc(void) {

	const float *pixels = film->GetScreenBuffer();

	glRasterPos2i(0, 0);

	glDrawPixels(film->GetWidth(), film->GetHeight(), GL_RGB, GL_FLOAT, pixels);

//	PrintCaptions();

	if (printHelp) {
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-0.5, 639.5, -0.5, 479.5, -1.0, 1.0);



		glPopMatrix();
	}

	glutSwapBuffers();
}

void reshapeFunc(int newWidth, int newHeight) {
	glViewport(0, 0, newWidth, newHeight);
	glLoadIdentity();
	glOrtho(0.f, newWidth - 1.0f, 0.f, newHeight - 1.0f, -1.f, 1.f);

	//config->ReInit(true, newWidth, newHeight);

	glutPostRedisplay();
}

#define MOVE_STEP 0.5f
#define ROTATE_STEP 4.f
void keyFunc(unsigned char key, int x, int y) {
	switch (key) {
		case 27: // Escape key
			cerr << "Done." << endl;
			exit(0);
			break;

		default:
			break;
	}

	displayFunc();
}

void specialFunc(int key, int x, int y) {
	switch (key) {

		default:
			break;
	}

	displayFunc();
}

static int mouseButton0 = 0;
static int mouseButton2 = 0;
static int mouseGrabLastX = 0;
static int mouseGrabLastY = 0;
static double lastMouseUpdate = 0.0;

static void mouseFunc(int button, int state, int x, int y) {
	if (button == 0) {
		if (state == GLUT_DOWN) {
			// Record start position
			mouseGrabLastX = x;
			mouseGrabLastY = y;
			mouseButton0 = 1;
		} else if (state == GLUT_UP) {
			mouseButton0 = 0;
		}
	} else if (button == 2) {
		if (state == GLUT_DOWN) {
			// Record start position
			mouseGrabLastX = x;
			mouseGrabLastY = y;
			mouseButton2 = 1;
		} else if (state == GLUT_UP) {
			mouseButton2 = 0;
		}
	}
}

static void motionFunc(int x, int y) {

}

void timerFunc(int value) {

}

void InitGlut(int argc, char *argv[], unsigned int width, unsigned int height) {
	glutInitWindowSize(width, height);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInit(&argc, argv);

	glutCreateWindow("SmallLuxGPU v1.3 (Written by David Bucciarelli)");
}

void RunGlut() {
	glutReshapeFunc(reshapeFunc);
	glutKeyboardFunc(keyFunc);
	glutSpecialFunc(specialFunc);
	glutDisplayFunc(displayFunc);
	glutMouseFunc(mouseFunc);
	glutMotionFunc(motionFunc);
	//glutTimerFunc(config->screenRefreshInterval, timerFunc, 0);

	glMatrixMode(GL_PROJECTION);
	glViewport(0, 0, film->GetWidth(), film->GetHeight());
	glLoadIdentity();
	glOrtho(0.f, film->GetWidth() - 1.f,
			0.f, film->GetHeight() - 1.f, -1.f, 1.f);

	glutMainLoop();
}
