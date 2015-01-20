#ifndef VOLUME_DATASET_H
#define VOLUME_DATASET_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "VoxelReader.h"
#include <string>
#include <time.h>
#include "GLM.h"
#include "Camera.h"



class VolumeDataset
{
public:
	GLubyte *memblock3D;
	GLuint currTexture3D;
	GLuint nextTexture3D;

	int timesteps;
	float timePerFrame;
	int numDims;
	int xRes, yRes, zRes;
	int bytesPerElement;
	bool littleEndian;
	std::string elementType;

	int currentTimestep;
	clock_t oldTime;

	VoxelReader voxelReader;

	void Init();
	void Update();
	void ReverseEndianness();
	GLuint GenerateTexture();
	void UpdateTexture();
};



#endif