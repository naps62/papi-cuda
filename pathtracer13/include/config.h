/*
 * config.h
 *
 *  Created on: Feb 13, 2012
 *      Author: rr
 */

#ifndef CONFIG_H_
#define CONFIG_H_



#define WIDTH 800
#define HEIGHT 600
#define MAX_PATH_DEPTH 5
#define SPP 1
#define SHADOWRAY 4

#define SCENE_FILE "scenes/luxball.scn"

#define PIXEL_COUNT (WIDTH * HEIGHT)
#define PATH_COUNT (PIXEL_COUNT * SPP)
#define SEED 54654654

#define CUDA

#endif /* CONFIG_H_ */
