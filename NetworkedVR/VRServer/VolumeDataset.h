#ifndef VOLUME_DATASET_H
#define VOLUME_DATASET_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VoxelReader.h"
#include <string>
#include <time.h>
#include "GLM.h"
#include "Camera.h"
#include <future>

#define NUM_STREAMING_THREADS 1

class VolumeDataset
{
public:
	GLubyte *threadBlock[NUM_STREAMING_THREADS];
	GLubyte *currMemblock;
	GLuint currTexture3D;

	int timesteps;
	float timePerFrame;
	int numDims;
	int xRes, yRes, zRes;
	int bytesPerElement;
	bool littleEndian;
	std::string elementType;
	int numVoxels;

	int currentTimestep;
	clock_t oldTime;
	std::future<void> asyncTexLoad[NUM_STREAMING_THREADS];

	VoxelReader voxelReader;

	void Init();
	void InitTexture();
	void Update();

	GLuint GenerateTexture();
	void UpdateTexture();
	void LoadTextureAsync(int currentThread, int stepsToBuffer);
	void CopyToTexture();

};



#endif