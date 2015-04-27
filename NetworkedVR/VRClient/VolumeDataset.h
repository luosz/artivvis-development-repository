#ifndef VOLUME_DATASET_H
#define VOLUME_DATASET_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VoxelReader.h"
#include <string>
#include <time.h>
#include "GLM.h"
#include "Camera.h"
#include "VoxelReader.h"

#define NUM_STREAMING_THREADS 2

class VolumeDataset
{
public:
	GLubyte *memblock3D;
	GLuint currTexture3D;

	int timesteps;
	float timePerFrame;
	int xRes, yRes, zRes;
	int bytesPerElement;
	bool littleEndian;
	int numVoxels;

	void Init();
	GLuint GenerateTexture();
	void CopyToTexture();
};



#endif