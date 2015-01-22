#ifndef CHANGE_BETWEEN_FRAMES_H
#define CHANGE_BETWEEN_FRAMES_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "CudaHeaders.h"
#include <vector>
#include "VolumeDataset.h"


class ChangeBetweenFrames
{
public:
	std::vector<cudaGraphicsResource_t> cudaResources;
	int texture3DSize;
	GLuint prevTexture3D;
	float *l1, *l2, *l3, *l4, *l5;
	float la1, la2, la3, la4, la5;

	void Init(VolumeDataset &volume);
	GLuint Generate3DTexture(VolumeDataset &volume);
	void Find(VolumeDataset &volume, int currentTimestep, GLuint bruteTex3D);
};


#endif