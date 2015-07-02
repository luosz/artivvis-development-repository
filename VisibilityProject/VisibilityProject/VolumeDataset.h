#ifndef VOLUME_DATASET_H
#define VOLUME_DATASET_H

#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VoxelReader.h"
#include <string>
#include <time.h>
#include "GLM.h"
#include "Camera.h"
#include <future>

#ifndef MAX_PATH
#define MAX_PATH          260
#endif

#define NUM_STREAMING_THREADS 2

class VolumeDataset
{
public:
	GLubyte *memblock3D;
	GLubyte *threadBlock[NUM_STREAMING_THREADS];
	GLubyte *currMemblock;
	GLuint currTexture3D;

	std::string folderPath;
	std::string headerFile;

	int timesteps;
	float timePerFrame;
	int numDims;
	int xRes, yRes, zRes;
	int bytesPerElement;
	bool littleEndian;
	std::string elementType;

	int currentTimestep;
	clock_t oldTime;
	std::future<void> asyncTexLoad[NUM_STREAMING_THREADS];

	VoxelReader voxelReader;

	void Init();
	void Update();
	void ReverseEndianness();
	void InitTexture();
	GLuint GenerateTexture();
	void UpdateTexture();
	void LoadTextureAsync(int currentThread, int stepsToBuffer);
	void CopyToTexture();
};

#endif
