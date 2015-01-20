#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VolumeDataset.h"

class BruteForce
{
public:
	GLuint tex3D;
	int textureSize;

	BruteForce(VolumeDataset &volume);
	GLuint GenerateTexture(VolumeDataset &volume);
	GLuint BruteForceCopy(VolumeDataset &volume, int currentTimestep);
};

#endif